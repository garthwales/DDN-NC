import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import torch
import os

from generate_dataset import texture_colour
from nets import EDNetwork
from utils import *

plt.rcParams.update({'font.size': 20})

# --------------------------------------------------------------------------------------------------------------------
# --- normalized cuts (2nd smallest eigenvector) ---
# --------------------------------------------------------------------------------------------------------------------

def NCExperiments(batch=10, iters=1000, trials=5, 
                  method='exact', loss_on='second_smallest', mat_type='psd', 
                  texture=True, output_folder='figures/'):
    """ Runs nc experiments """

    learning_curves = [[] for i in range(trials)]
    cosine_sim_curves = [[] for i in range(trials)]

    # TODO: timing and memory trials for these specific ones...
    texture_colour('data/', batch) # generate the images if needed
    
    size=(28,28)
    
    device = torch.device(f'cuda:1' if torch.cuda.is_available() else 'cpu') 
    # or f'cuda:{args.gpu}' or 'cuda:1'
    
    if True:
        X_input = load_images_from_directory('data/tc/img/', num=batch, size=size)
        if texture:
            Q_true = load_images_from_directory('data/tc/maskT', num=batch, size=size)
        else:
            Q_true = load_images_from_directory('data/tc/maskC', num=batch, size=size)
    else:
        # TODO: bw experiment here
        return
    
    
    
    dim_z = X_input.shape[1] * X_input.shape[2] 
    m = dim_z
    
    Q_true = Q_true.flatten(start_dim=1) # Flatten to match flat outputs...
    X_input = X_input.flatten(start_dim=1) # Flatten to match linear layers? IDK
        
    
    print(device)
    print(f'{dim_z}-{m}-{method}-{mat_type}')
    
    # TODO: change top_k if second smallest is selected..
    # TODO: change top_k if smallest is selected..?
    # TODO: test max without top_k set?
    
    model = EDNetwork(dim_z, m, method=method, matrix_type=mat_type)
    
    with torch.cuda.device(device):
        # torch.cuda.empty_cache()
        X_input = X_input.to(device)
        Q_true = Q_true.to(device)
            
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    
    for trial in range(trials):
        # prepare data and model
        torch.manual_seed(22 + trial)

        # do optimisation
        for i in range(iters):
            optimizer.zero_grad(set_to_none=True)
            try:
                Q_pred = model(X_input)
            except KeyboardInterrupt:
                torch.save(model.state_dict(), 'INTERRUPTED.pth')
                print('Saved interrupt to INTERRUPTED.pth')
                raise
            except:
                break                
            
            if loss_on == 'second_smallest':
                # NOTE: for the stuff from anu it would be Q_true[:,:,1] as it is assuming output is including everything
                loss = 1.0 - torch.mean(torch.abs(torch.nn.functional.cosine_similarity(Q_pred[:, :, 1], Q_true))) # second smallest
                if i % 100 == 0:
                    name = f'trial-{trial}-iter-{i}.png'
                    save_plot_imgs(Q_pred[:,:,1].reshape((-1,28,28)).detach().cpu().numpy(), output_name=name, output_path=output_folder)
            else:
                assert False, "loss_on must be one of ('all', 'max', 'min', 'second_smallest)" 
            learning_curves[trial].append(float(loss.item()))
            loss.backward()
            optimizer.step()

            if (i % 100 == 0):
                print("{: 3} {: 6}: {}".format(trial, i, loss.item()))

    return learning_curves, cosine_sim_curves


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    print(torch.__version__)
    print(torch.cuda.get_device_name() if torch.cuda.is_available() else "No CUDA")

    enable_ed_random_exp = True
    # -- sub experiments vvvvvv
    exact_exp            = True
    pytorch_exp          = False

    # --------------------------------------------------------------------------------------------------------------------
    # --- normalized cuts ---
    # --------------------------------------------------------------------------------------------------------------------

    prefix='tc/'
    base_dir = 'figures'
    os.makedirs(base_dir, exist_ok=True)
    date_string = datetime.now().strftime('%Y%m%d-%H%M%S')
    
    batch=100
    iters=2000
    trials=1
    method='exact'
    loss_on='second_smallest'
    mat_type='psd'
    # texture=True # the variable for this experiment

    if enable_ed_random_exp:
        if exact_exp:
            option = 'exact/'
            save_dir = os.path.join(base_dir, date_string, prefix, option)
            os.makedirs(save_dir, exist_ok=True)
            
            # TODO: here or in experiements.. but write a txt with all the params included..
            
            # second smallest ev, psd matrix, texture labels
            exact_curves, _ = NCExperiments(batch=batch, iters=iters, trials=trials,
                                            method='exact', loss_on='second_smallest', 
                                            mat_type='psd', texture=True,
                                            output_folder=save_dir)
        
            plt.figure()
            PlotResults(exact_curves, fcn=plt.plot)
            plt.savefig(save_dir+f'{mat_type}_texture.pdf', dpi=300, bbox_inches='tight')
            
            # second smallest ev, psd matrix, color labels
            exact_curves, _ = NCExperiments(batch=batch, iters=iters, trials=trials,
                                            method='exact', loss_on='second_smallest', 
                                            mat_type='psd', texture=False,
                                            output_folder=save_dir)
        
            plt.figure()
            PlotResults(exact_curves, fcn=plt.plot)
            plt.savefig(save_dir+f'{mat_type}_colour.pdf', dpi=300, bbox_inches='tight')
        
        if pytorch_exp:
            option = 'torch/'
            save_dir = os.path.join(base_dir, date_string, prefix, option)
            os.makedirs(save_dir, exist_ok=True)
            
            # second smallest ev, psd matrix, texture labels
            exact_curves, _ = NCExperiments(batch=batch, iters=iters, trials=trials,
                                            method='pytorch', loss_on='second_smallest', 
                                            mat_type='psd', texture=True,
                                            output_folder=save_dir)

            plt.figure()
            PlotResults(exact_curves, fcn=plt.plot)
            plt.savefig(save_dir+f'{mat_type}_texture.pdf', dpi=300, bbox_inches='tight')
            
            # second smallest ev, psd matrix, color labels
            exact_curves, _ = NCExperiments(batch=batch, iters=iters, trials=trials,
                                            method='pytorch', loss_on='second_smallest', 
                                            mat_type='psd', texture=False,
                                            output_folder=save_dir)

            plt.figure()
            PlotResults(exact_curves, fcn=plt.plot)
            plt.savefig(save_dir+f'{mat_type}_colour.pdf', dpi=300, bbox_inches='tight')
