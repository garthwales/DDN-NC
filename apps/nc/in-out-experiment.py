import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 20})

import torch, os
import torch.nn as nn

os.makedirs("figures", exist_ok=True)

def PlotResults(exact_curves, fcn=plt.semilogy):
    """plot results of experiments."""
    exact_mean = np.mean(exact_curves, axis=0)

    fcn(exact_mean, 'b')
    for trial in range(len(exact_curves)):
        fcn(exact_curves[trial], 'b', alpha=0.1)
    fcn(exact_mean, 'b')
    plt.xlabel('iter.')
    plt.ylabel('loss')

# --------------------------------------------------------------------------------------------------------------------
# --- eigen decomposition ---
# --------------------------------------------------------------------------------------------------------------------

from eigen import EigenDecompositionFcn

class EDNetwork(nn.Module):
    """Example eigen decomposition network comprising a MLP data processing layer followed by a
    differentiable eigen decomposition layer. Input is (B, Z, 1); output is (B, M, M)."""

    def __init__(self, dim_z, m, method='exact', top_k=None, matrix_type='psd'):
        super(EDNetwork, self).__init__()

        self.dim_z = dim_z
        self.m = m
        self.method = method
        self.top_k = None # TODO: use this (or something similar) to test further..
        self.matrix_type = matrix_type

        self.mlp = nn.Sequential(
            nn.Linear(dim_z, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 20),
            nn.ReLU(),
            nn.Linear(20, m * m)
        )

    def forward(self, z):
        # construct input for declarative node
        z = z.flatten(start_dim=1) # bxnxn -> bxdim_z
        # assert z.shape[1] == self.dim_z
        
        x = self.mlp(z)
        x = torch.reshape(x, (z.shape[0], self.m, self.m))

        if self.matrix_type == 'general':
            pass
        elif self.matrix_type == 'psd':
            x = torch.matmul(x, x.transpose(1, 2)) # positive definite
        elif self.matrix_type == 'rank1':
            u = x
            x = torch.matmul(u[:, :, 0], u[:, :, 0].transpose(1, 2))
        else:
            assert False, "unknown matrix_type"

        if self.method == 'pytorch':
            x = 0.5 * (x + x.transpose(1, 2))
            v, y = torch.linalg.eigh(x)
        elif self.method == 'exact':
            y = EigenDecompositionFcn().apply(x, self.top_k)
        else:
            assert False

        return y

# Function to load all images from a directory and convert to PyTorch tensors
def load_images_from_directory(directory, num=10, size=(28,28)):
    import cv2
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    images = []
    for i, filename in enumerate(os.listdir(directory)):
        if i >= num:
            return torch.stack(images)
        img_path = os.path.join(directory, filename)
        image = cv2.resize(cv2.imread(img_path,0), size)
        image = transform(image).squeeze() # ToTensor() adds too much
        images.append(image)
    return torch.stack(images)

def add_headers(
    fig,
    *,
    row_headers=None,
    col_headers=None,
    row_pad=1,
    col_pad=5,
    rotate_row_headers=True,
    **text_kwargs
):
    # Based on https://stackoverflow.com/a/25814386

    axes = fig.get_axes()

    for ax in axes:
        sbs = ax.get_subplotspec()

        # Putting headers on cols
        if (col_headers is not None) and sbs.is_first_row():
            ax.annotate(
                col_headers[sbs.colspan.start],
                xy=(0.5, 1),
                xytext=(0, col_pad),
                xycoords="axes fraction",
                textcoords="offset points",
                ha="center",
                va="baseline",
                **text_kwargs,
            )

        # Putting headers on rows
        if (row_headers is not None) and sbs.is_first_col():
            ax.annotate(
                row_headers[sbs.rowspan.start],
                xy=(0, 0.5),
                xytext=(-ax.yaxis.labelpad - row_pad, 0),
                xycoords=ax.yaxis.label,
                textcoords="offset points",
                ha="right",
                va="center",
                rotation=rotate_row_headers * 90,
                **text_kwargs,
            )

def plot_images(imgs, labels=None, row_headers=None, col_headers=None, colmns=None, title=None):
    """
    imgs = [img1, img2]
    labels = ['label1', 'label2']
    colmns = 2 (so will be a 1x2 size display)
    """
    num = len(imgs)
    # Calculate the given number of subplots, or use colmns count to get a specific output
    if colmns is None:
        ay = np.ceil(np.sqrt(num)).astype(int) # this way it will prefer rows rather than columns
        ax = np.rint(np.sqrt(num)).astype(int)
    else:
        ax = np.ceil(num / colmns).astype(int)
        ay = colmns
        
    fig = plt.figure()
    if title is not None:
        fig.suptitle(title, fontsize=16)
    
    for i in range(1, num+1):
        sub = fig.add_subplot(ax,ay,i)
        if labels is not None:
            sub.set_title(f'{labels[i-1]}')
            
        sub.axis('off')
        sub.imshow(imgs[i-1])
        
    add_headers(fig, row_headers=row_headers, col_headers=col_headers, rotate_row_headers=False)

def EDExperiments(dim_z=20, m=20, batch=10, iters=200, trials=10, method='pytorch', loss_on='second_smallest', mat_type='psd', texture=True):
    from generate_dataset import texture_colour
    # from torchsummary import summary
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    

    learning_curves = [[] for i in range(trials)]
    cosine_sim_curves = [[] for i in range(trials)]

    # TODO: timing and memory trials for these specific ones...
    texture_colour('data/', batch) # generate the images if needed
    
    size=(28,28)
    
    if True:
        X_input = load_images_from_directory('data/tc/img/', num=batch, size=size).cuda()
        if texture:
            Q_true = load_images_from_directory('data/tc/maskT', num=batch, size=size)
        else:
            Q_true = load_images_from_directory('data/tc/maskC', num=batch, size=size)
    else:
        # TODO: bw experiment here
        return
    
    Q_true = Q_true.flatten(start_dim=1).cuda() # Flatten to match flat outputs...

    for trial in range(trials):
        # prepare data and model
        torch.manual_seed(22 + trial)

        dim_z = X_input.shape[1] * X_input.shape[2] 
        m = dim_z
        # NOTE: X_input is flattened within forward, but not here for visualisation purposes
        
        model = EDNetwork(dim_z, m, method=method, top_k=1 if loss_on=='max' else None, matrix_type=mat_type).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
        
        # summary(model, X_input.shape)

        # TODO: change top_k if second smallest is selected..
        # TODO: change top_k if smallest is selected..?
        # TODO: test max without top_k set?

        # do optimisation
        for i in range(iters):
            optimizer.zero_grad(set_to_none=True)
            Q_pred = model(X_input)
            if loss_on == 'all':
                loss = 1.0 - torch.mean(torch.abs(torch.nn.functional.cosine_similarity(Q_pred, Q_true, dim=1))) # all ev
            elif loss_on == 'max':
                loss = 1.0 - torch.mean(torch.abs(torch.nn.functional.cosine_similarity(Q_pred[:, :, -1], Q_true[:, :, -1]))) # largest
            elif loss_on == 'min':
                loss = 1.0 - torch.mean(torch.abs(torch.nn.functional.cosine_similarity(Q_pred[:, :, 0], Q_true[:, :, 0]))) # smallest
            elif loss_on == 'second_smallest':
                # NOTE: for the stuff from anu it would be Q_true[:,:,1] as it is assuming output is including everything
                loss = 1.0 - torch.mean(torch.abs(torch.nn.functional.cosine_similarity(Q_pred[:, :, 1], Q_true))) # second smallest
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
    pytorch_exp          = True

    # --------------------------------------------------------------------------------------------------------------------
    # --- eigen decomposition ---
    # --------------------------------------------------------------------------------------------------------------------

    # Defaults to 10 batch of 20x20 weights matrix

    prefix='tc'

    if enable_ed_random_exp:
        if exact_exp:
            # second smallest ev, psd matrix, texture labels
            exact_curves, _ = EDExperiments(method='exact', loss_on='second_smallest', mat_type='psd', texture=True)
        
            plt.figure()
            PlotResults(exact_curves, fcn=plt.plot)
            plt.savefig('figures/'+prefix+'_exact_second_psd_tex.pdf', dpi=300, bbox_inches='tight')
            
            # second smallest ev, psd matrix, color labels
            exact_curves, _ = EDExperiments(method='exact', loss_on='second_smallest', mat_type='psd', texture=False)
        
            plt.figure()
            PlotResults(exact_curves, fcn=plt.plot)
            plt.savefig('figures/'+prefix+'_exact_second_psd_col.pdf', dpi=300, bbox_inches='tight')
        
        if pytorch_exp:
            # second smallest ev, psd matrix, texture labels
            exact_curves, _ = EDExperiments(method='pytorch', loss_on='second_smallest', mat_type='psd', texture=True)

            plt.figure()
            PlotResults(exact_curves, fcn=plt.plot)
            plt.savefig('figures/'+prefix+'_torch_second_psd_tex.pdf', dpi=300, bbox_inches='tight')
            
            # second smallest ev, psd matrix, color labels
            exact_curves, _ = EDExperiments(method='pytorch', loss_on='second_smallest', mat_type='psd', texture=True)

            plt.figure()
            PlotResults(exact_curves, fcn=plt.plot)
            plt.savefig('figures/'+prefix+'_torch_second_psd_tex.pdf', dpi=300, bbox_inches='tight')
