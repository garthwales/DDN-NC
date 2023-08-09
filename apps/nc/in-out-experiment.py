import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import os

import wandb # TODO: use this..

from utils.generate_dataset import texture_colour
from nets.nets import GenericNC, BasicMLP
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
    texture_colour('data/', args.batch) # generate the images if needed
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu') 
    print(device)
    
    X_input = load_images_from_directory(args.dir_input, num=args.batch, size=args.size, gray=args.grayscale)
    Q_true = load_images_from_directory(args.dir_output, num=args.batch, size=args.size, gray=True)
    
    # TODO: change top_k if second smallest is selected..
    # TODO: change top_k if smallest is selected..?
    # TODO: test max without top_k set?
    
    Q_true = Q_true.flatten(start_dim=1) # Flatten to match flat outputs...
    
    channels = 1 if args.grayscale else 3
    
    # these could have non-hardcoded sizes but meh
    if args.net == 'UNet':
        pre_net = UNet(channels, args.size[0]*args.size[1])
    elif args.net == 'resnet':
        # pre_net = ResNetBase()
        assert 'not implement yet'
    elif args.net == 'MLP':
        pre_net = BasicMLP(channels*args.size[0]*args.size[1], args.size[0])
        X_input = X_input.flatten(start_dim=1) # Flatten to match linear layers? IDK
        
    model = GenericNC(pre_net, args.size[0], args.net, args.mat_type, args.method)
    
    if torch.cuda.is_available():
        with torch.cuda.device(device):
            # torch.cuda.empty_cache()
            X_input = X_input.to(device)
            Q_true = Q_true.to(device)
            model = model.to(device)
            
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    for trial in range(args.trials):
        # prepare data and model
        torch.manual_seed(22 + trial)

        # do optimisation
        for i in range(args.iters):
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
                if i % 100 == 0:
                    name = f'trial-{trial}-iter-{i}.png'
                    save_plot_imgs(Q_pred[:,:,1].reshape((-1,28,28)).detach().cpu().numpy()[0:5], output_name=name, output_path=output_folder)
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
        gpu=1,
        batch=50, 
        iters=2000, 
        trials=5, 
        method='exact',
        mat_type='psd',
        loss_on='second_smallest',
        net='UNet',
        
        size = (28,28),
        lr=1e-4,
        grayscale=False,
        
        dir_input = 'data/tc/img/',
        dir_output = 'data/tc/maskT',
        seed = 0
    )
    
    experiment = wandb.init(project='DDN-NC', config=args, mode='disabled')
    # Config parameters are automatically set by W&B sweep agent
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
            option = 'exact/'
            save_dir = os.path.join(base_dir, date_string, prefix, option)
            os.makedirs(save_dir, exist_ok=True)
            
            # TODO: here or in experiements.. but write a txt args saved
            # TODO: add wandbai to this..
            
            # second smallest ev, psd matrix, texture labels
            exact_curves, _ = NCExperiments(args, output_folder=save_dir)
        
            plt.figure()
            PlotResults(exact_curves, fcn=plt.plot)
            plt.savefig(save_dir+f'{args.mat_type}_texture.pdf', dpi=300, bbox_inches='tight')
            
            # second smallest ev, psd matrix, color labels
            args.dir_output = 'data/tc/maskT'
            exact_curves, _ = NCExperiments(args, output_folder=save_dir)
        
            plt.figure()
            PlotResults(exact_curves, fcn=plt.plot)
            plt.savefig(save_dir+f'{args.mat_type}_colour.pdf', dpi=300, bbox_inches='tight')