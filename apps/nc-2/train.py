# import argparse
# import logging
import os
import torch
from pathlib import Path
from torch import optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from pprint import pprint

import wandb
from evaluate import evaluate

# import each different model
from nets.unet import UNet
from nets.resnet import ResNetBase

from nets.nets import GenericNC, BasicMLP, BasicCNN

from utils.data_loading import TwoFolders

def train_model(model, device, args):
    # 1. Create dataset
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    dataset = TwoFolders(args.dir_img, args.dir_mask, transform, args.size)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * args.val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    # 3. Create data loaders
    loader_args = dict(batch_size=args.batch_size, num_workers=os.cpu_count()-1, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=False, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    pprint(args)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.patience)  # goal: minimize loss
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    
    global_step = 0
    
    torch.cuda.empty_cache()

    wandb.watch(model, log_freq=100)
    
    # 5. Begin training
    for epoch in range(1, args.epochs + 1):
        model.train()
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{args.epochs}', unit='img', position=0, leave=False) as pbar:
            for images, true_masks in train_loader:
                images = images.to(device=device) # memory_format=torch.channels_last
                true_masks = true_masks.to(device=device)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=args.amp):
                    masks_pred = model(images)
                    loss = 1.0 - torch.mean(torch.abs(torch.nn.functional.cosine_similarity(masks_pred[:, :, 1], true_masks.view(images.shape[0],-1))))

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                if args.gradient_clipping is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                wandb.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch,
                    'min_pred': torch.min(masks_pred).item(),
                    'max_pred': torch.max(masks_pred).item(),
                    'avg_pred': torch.mean(masks_pred).item(),
                    
                    # 'min_true': torch.min(true_masks).item(),
                    # 'max_true': torch.max(true_masks).item(),
                    # 'avg_true': torch.mean(true_masks).item(),
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                # division_step = (n_train // (5 * args.batch_size))
                # if division_step > 0:
                if global_step % 5 == 0 or global_step == 1:
                    val_score = evaluate(model, val_loader, device, args.amp)
                    scheduler.step(val_score)

                    # print('Validation loss: {}'.format(val_score))
                    wandb.log({
                        'learning rate': optimizer.param_groups[0]['lr'],
                        'val': val_score,
                        'images': wandb.Image(images[0].cpu()),
                        'masks': {
                            'true': wandb.Image(true_masks[0].cpu().float()),
                            'pred': wandb.Image(masks_pred[:,:,1].view(images.shape[0],args.size[0], args.size[1])[0].cpu()),
                        },
                        'step': global_step,
                        'epoch': epoch,
                    })
                        # **histograms
                        
                    # Log NCutCost (obj and const)
                    # Log Eigen solution value? (e.g. the other potential obj funcs)
                    # Log Histogram of pred values

        if args.save_checkpoint:
            Path(args.dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(args.dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            print(f'Checkpoint {epoch} saved!')

if __name__ == '__main__':
    
    defaults_dict = dict(
        epochs=100, 
        batch_size = 10,
        val_percent=0.1,

        lr = 1e-3, # will lower during training with patience
        patience=5,

        dir_img = 'data/samples/img2',
        dir_mask = 'data/samples/tex2',

        grayscale=False,
        method='exact',
        mat_type='general',
        loss_on='second_smallest',
        net='UNet',
        net_name='UNet',
        
        size = (48,48),
        width = -1,
        laplace = None,

        seed=10, # TODO: loop through different seeds to produce different trial runs
        # gpu=1,
        # img_scale=1, # Downscaling factor of the images

        gradient_clipping = 1,
        save_checkpoint= False,
        dir_checkpoint='',
        
        load=False, # Load model from a .pth file
        # Not likely to use for IVCNZ at least
        amp=False, # Use mixed precision
        # Configuration does nothing, but important to note
        optim='AdamW',
        # shuffle=False,
        )

    wandb.init(project='IVCNZ', config=defaults_dict) # mode='disabled'
    # Config parameters are automatically set by W&B sweep agent
    args = wandb.config
     
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    print(f'Using device {device}')
    
    channels = 1 if args.grayscale else 3

    if 'only' in args.net:
        if args.net == 'UNet-only':
            model = UNet(channels, 1)
    else:
        if args.net == 'UNet':
            pre_net = UNet(channels, args.size[0]*args.size[1])
        elif args.net == 'resnet':
            pre_net = ResNetBase()
            assert 'not implement yet'
        elif args.net == 'MLP':
            # X_input = X_input.flatten(start_dim=1) # Flatten to match linear layers? IDK
            pre_net = BasicMLP(channels*args.size[0]*args.size[1], args.size[0])
        elif args.net == 'cnn':
            pre_net = BasicCNN(args.size[1], args.size[0]*args.size[0]*args.size[0]*args.size[0])
        elif args.net == 'vgg':
            assert 'not implemented yet'
        else:
            assert 'provide a valid net (UNet, resnet, MLP, cnn)'
    
            
        model = GenericNC(pre_net, args.size[0], args.net_name, args.mat_type, args.method)
    

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        print(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(model, device, args)
    except torch.cuda.OutOfMemoryError:
        print('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(model, device, args)
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        raise
