import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import os

import wandb # TODO: use this..

from nets.nets import GenericNC, BasicMLP, BasicCNN
from nets.unet import UNet
from nets.resnet import ResNetBase
from utils.utils import load_images_from_directory, PlotResults, save_plot_imgs

plt.rcParams.update({'font.size': 20})

# --------------------------------------------------------------------------------------------------------------------
# --- normalized cuts (2nd smallest eigenvector) ---
# --------------------------------------------------------------------------------------------------------------------

def NCExperiments(args, output_folder):
    """ Runs nc experiments """

    learning_curves = [[] for i in range(args.trials)]
    cosine_sim_curves = [[] for i in range(args.trials)]

    # TODO: timing and memory trials for these specific ones...
    # texture_colour('data/', args.batch) # generate the images if needed
    
    if args.gpu is not None:
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu') 
    else:
        device = torch.device('cpu')
    print(device)
    
    X_input = load_images_from_directory(args.dir_input, num=args.batch, size=args.size, gray=args.grayscale)
    Q_true = load_images_from_directory(args.dir_output, num=args.batch, size=args.size, gray=True)
    
    # save_plot_imgs(np.moveaxis(X_input.numpy(), 1, -1), output_name='X_true.png', output_path=output_folder)
    
    # TODO: change top_k if second smallest is selected..
    # TODO: change top_k if smallest is selected..?
    # TODO: test max without top_k set?
    
    Q_true = Q_true.flatten(start_dim=1) # Flatten to match flat outputs...
    
    channels = 1 if args.grayscale else 3
    
    # these could have non-hardcoded sizes but meh
    if args.net == 'UNet':
        pre_net = UNet(channels, args.size[0]*args.size[1])
    elif args.net == 'resnet':
        pre_net = ResNetBase()
        assert 'not implement yet'
    elif args.net == 'MLP':
        X_input = X_input.flatten(start_dim=1) # Flatten to match linear layers? IDK
        pre_net = BasicMLP(channels*args.size[0]*args.size[1], args.size[0])
    elif args.net == 'cnn':
        pre_net = BasicCNN(args.size[1], args.width*args.size[0]*args.size[0])
    elif args.net == 'vgg':
        assert 'not implemented yet'
    else:
        assert 'provide a valid net (UNet, resnet, MLP, cnn)'
        
    
    if torch.cuda.is_available() and args.gpu is not None:
        with torch.cuda.device(device):
            # torch.cuda.empty_cache()
            X_input = X_input.to(device)
            Q_true = Q_true.to(device)
            
    for trial in range(args.trials):
        # prepare data and model
        torch.manual_seed(22 + trial)

        # reset model for each trial
        model = GenericNC(pre_net, args.size[0], args.net, args.mat_type, args.method, args.width, args.laplace)
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        # do optimisation
        for i in range(args.iters+1): # +1 so the plotting will happen for the end number as well
            optimizer.zero_grad(set_to_none=True)
            try:
                Q_pred = model(X_input)
            except KeyboardInterrupt:
                torch.save(model.state_dict(), 'INTERRUPTED.pth')
                print('Saved interrupt to INTERRUPTED.pth')
                raise
            # except Exception as err:
            #     print(f'{err} on trial {trial}, iter {i}')
            #     break                
            
            if args.loss_on == 'second_smallest':
                # NOTE: for the stuff from anu it would be Q_true[:,:,1] as it is assuming output is including everything
                loss = 1.0 - torch.mean(torch.abs(torch.nn.functional.cosine_similarity(Q_pred[:, :, 1], Q_true))) # second smallest
                if i % 10 == 0:
                    name = f'{args.out_type}-trial-{trial}-iter-{i}.png'
                    save_plot_imgs(Q_pred[:,:,1].reshape((-1,args.size[0],args.size[1])).detach().cpu().numpy(), output_name=name, output_path=output_folder)
            else:
                assert False, "loss_on must be one of ('all', 'max', 'min', 'second_smallest)" 
            learning_curves[trial].append(float(loss.item()))
            loss.backward()
            optimizer.step()

            if (i % 100 == 0):
                print("{: 3} {: 6}: {}".format(trial, i, loss.item()))

    return learning_curves, cosine_sim_curves


if __name__ == '__main__':
    args = dict(
        gpu=0,
        batch=2, 
        iters=1000, 
        trials=3, 
        method='exact',
        mat_type='general',
        loss_on='second_smallest',
        net='cnn',
        
        size = (96,96),
        width = 50,
        laplace = None,
        
        lr=1e-3,
        grayscale=False,
        
        dir_input = 'data/samples/img/',
        dir_output = 'data/samples/col/',
        out_type = 'colour',
        seed = 0
    )
    
    experiment = wandb.init(project='DDN-NC', config=args, mode='disabled', allow_val_change=True)
    # disabled and allow val changes... otherwise enable and use sweep to set different params
    args = wandb.config
    
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    print(torch.__version__)
    print(torch.cuda.get_device_name() if torch.cuda.is_available() else "No CUDA")

    enable_ed_random_exp = True
    # -- sub experiments vvvvvv
    exact_exp            = True

    # --------------------------------------------------------------------------------------------------------------------
    # --- normalized cuts ---
    # --------------------------------------------------------------------------------------------------------------------

    prefix='tc/'
    base_dir = 'figures'
    os.makedirs(base_dir, exist_ok=True)
    date_string = datetime.now().strftime('%Y%m%d-%H%M%S')
    
    if enable_ed_random_exp:
        if exact_exp:
            args.dir_output = 'data/samples/col/'
            args.out_type = 'colour'
            save_dir = os.path.join(base_dir, date_string, prefix)
            os.makedirs(save_dir, exist_ok=True)
            
            # TODO: here or in experiements.. but write a txt args saved
            # TODO: add wandbai to this..
            
            # second smallest ev, psd matrix, texture labels
            exact_curves, _ = NCExperiments(args, output_folder=save_dir)
        
            plt.figure()
            PlotResults(exact_curves, fcn=plt.plot)
            plt.savefig(save_dir+f'{args.mat_type}_{args.out_type}.pdf', dpi=300, bbox_inches='tight')
            
            # second smallest ev, psd matrix, color labels
            args.dir_output = 'data/samples/tex/'
            args.out_type = 'texture'
            exact_curves, _ = NCExperiments(args, output_folder=save_dir)
        
            plt.figure()
            PlotResults(exact_curves, fcn=plt.plot)
            plt.savefig(save_dir+f'{args.mat_type}_{args.out_type}.pdf', dpi=300, bbox_inches='tight')
