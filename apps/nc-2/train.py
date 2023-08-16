import argparse
# import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from pprint import pprint

import wandb
from evaluate import evaluate

# import each different model
from nets.unet import UNet as UNet
from nets.resnet import ResNetBase

from nets.nets import GenericNC, BasicMLP

from utils.data_loading import BasicDataset

def train_model(model, device, args):
    # 1. Create dataset
    dataset = BasicDataset(args.dir_img, args.dir_mask, args.img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * args.val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=args.batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    pprint(args)

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    # optimizer = optim.RMSprop(model.parameters(),
    #                           lr=args.lr, weight_decay=weight_decay, momentum=momentum, foreach=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)  # goal: minimize loss
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    
    global_step = 0

    # 5. Begin training
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{args.epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=args.amp):
                    masks_pred = model(images)
                    loss = 1.0 - torch.mean(torch.abs(torch.nn.functional.cosine_similarity(masks_pred[:, :, 1], true_masks)))

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * args.batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(model, val_loader, device, args.amp)
                        scheduler.step(val_score)

                        print('Validation Dice score: {}'.format(val_score))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass

        if args.save_checkpoint:
            Path(args.dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(args.dir_checkpoint / 'checkpoint_epoch{}.pth'.format(epoch)))
            print(f'Checkpoint {epoch} saved!')

if __name__ == '__main__':
    
    defaults_dict = dict(
        net_name = 'unet',
        epochs=100, 
        batch_size = 100,

        lr = 1e-3, # will lower during training with patience
        patience=10,

        dir_img = 'data/tc/img/',
        dir_mask = 'data/tc/maskC',

        val=10, # Percent of the data that is used as validation (0-100)

        n_classes=784, # Number of classes
        n_channels=3, # 3 for RGB inputs

        seed=22, # TODO: loop through different seeds to produce different trial runs
        gpu=1,
        img_scale=1, # Downscaling factor of the images

        mat_type='psd',
        method='exact',

        gradient_clipping = 1.0,
        
        load=False, # Load model from a .pth file
        # Not likely to use for IVCNZ at least
        amp=False, # Use mixed precision
        # Configuration does nothing, but important to note
        optim='adam',
        shuffle=True)

    experiment = wandb.init(project='DDN-NC', config=defaults_dict)
    # Config parameters are automatically set by W&B sweep agent
    args = wandb.config
     
    # logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    if args.net_name == 'unet':
        pre_net = UNet(args.n_channels,args.n_classes) 
    elif args.net_name == 'mlp':
        pre_net = BasicMLP(3*28*28, 28*28)
    else:
        assert 'provide a valid net_name (unet, mlp)'
        
    model = GenericNC(pre_net, 28, args.net_name, args.mat_type, args.method)
    
    # NOTE: Not sure if this memory format does anything useful?
    model = model.to(memory_format=torch.channels_last) 

    print(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        print(f'Model loaded from {args.load}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except torch.cuda.OutOfMemoryError:
        print('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            device=device,
            img_scale=args.scale,
            val_percent=args.val / 100,
            amp=args.amp
        )
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        raise
