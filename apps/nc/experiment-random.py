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
            nn.Linear(dim_z, m * m),
            nn.ReLU(),
            nn.Linear(m * m, m * m),
            nn.ReLU(),
            nn.Linear(m * m, m * m)
        )

    def forward(self, z):
        assert z.shape[1] == self.dim_z

        # construct input for declarative node
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


def EDExperiments(dim_z=5, m=10, batch=10, iters=200, trials=10, method='exact', loss_on='all', mat_type='general', rand_in=False):

    learning_curves = [[] for i in range(trials)]
    cosine_sim_curves = [[] for i in range(trials)]

    # TODO: timing and memory trials for these specific ones...

    for trial in range(trials):
        # prepare data and model
        torch.manual_seed(22 + trial)
        X_true = torch.rand((batch, m, m), dtype=torch.float, requires_grad=False)
        if not rand_in: # make it psd instead of random
            X_true = torch.matmul(X_true, X_true.transpose(1, 2))
            # TODO: double check this against 0.5 * X + X.T sorta dealio
        V_true, Q_true = torch.linalg.eigh(X_true)
        z_init = torch.randn((batch, dim_z), dtype=torch.float, requires_grad=False)

        model = EDNetwork(dim_z, m, method=method, top_k=1 if loss_on=='max' else None, matrix_type=mat_type)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)

        # TODO: change top_k if second smallest is selected..
        # TODO: change top_k if smallest is selected..?
        # TODO: test max without top_k set?

        # do optimisation
        for i in range(iters):
            optimizer.zero_grad(set_to_none=True)
            Q_pred = model(z_init)
            if loss_on == 'all':
                loss = 1.0 - torch.mean(torch.abs(torch.nn.functional.cosine_similarity(Q_pred, Q_true, dim=1))) # all ev
            elif loss_on == 'max':
                loss = 1.0 - torch.mean(torch.abs(torch.nn.functional.cosine_similarity(Q_pred[:, :, -1], Q_true[:, :, -1]))) # largest
            elif loss_on == 'min':
                loss = 1.0 - torch.mean(torch.abs(torch.nn.functional.cosine_similarity(Q_pred[:, :, 0], Q_true[:, :, 0]))) # smallest
            elif loss_on == 'second_smallest':
                loss = 1.0 - torch.mean(torch.abs(torch.nn.functional.cosine_similarity(Q_pred[:, :, 1], Q_true[:, :, 1]))) # second smallest
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

    if enable_ed_random_exp:
        if exact_exp:
            # all evs, psd matrix, random input
            exact_curves, _ = EDExperiments(dim_z=5, method='exact', loss_on='all', mat_type='psd', rand_in=True)

            plt.figure()
            PlotResults(exact_curves, fcn=plt.plot)
            plt.savefig('figures/exact_all_psd_rand.pdf', dpi=300, bbox_inches='tight')
            
            # second smallest ev, psd matrix, random input
            exact_curves, _ = EDExperiments(dim_z=5, method='exact', loss_on='second_smallest', mat_type='psd', rand_in=True)

            plt.figure()
            PlotResults(exact_curves, fcn=plt.plot)
            plt.savefig('figures/exact_second_psd_rand.pdf', dpi=300, bbox_inches='tight')

            # min ev, psd matrix, random input
            exact_curves, _ = EDExperiments(dim_z=5, method='exact', loss_on='min', mat_type='psd', rand_in=True)

            plt.figure()
            PlotResults(exact_curves, fcn=plt.plot)
            plt.savefig('figures/exact_min_psd_rand.pdf', dpi=300, bbox_inches='tight')

            # max ev, psd matrix, random input
            exact_curves, _ = EDExperiments(dim_z=5, method='exact', loss_on='max', mat_type='psd', rand_in=True)

            plt.figure()
            PlotResults(exact_curves, fcn=plt.plot)
            plt.savefig('figures/exact_max_psd_rand.pdf', dpi=300, bbox_inches='tight')
            
            # all evs, psd matrix, psd input
            exact_curves, _ = EDExperiments(dim_z=5, method='exact', loss_on='all', mat_type='psd', rand_in=False)

            plt.figure()
            PlotResults(exact_curves, fcn=plt.plot)
            plt.savefig('figures/exact_all_psd.pdf', dpi=300, bbox_inches='tight')
            
            # second smallest ev, psd matrix, psd input
            exact_curves, _ = EDExperiments(dim_z=5, method='exact', loss_on='second_smallest', mat_type='psd', rand_in=False)

            plt.figure()
            PlotResults(exact_curves, fcn=plt.plot)
            plt.savefig('figures/exact_second_psd.pdf', dpi=300, bbox_inches='tight')

            # min ev, psd matrix, psd input
            exact_curves, _ = EDExperiments(dim_z=5, method='exact', loss_on='min', mat_type='psd', rand_in=False)

            plt.figure()
            PlotResults(exact_curves, fcn=plt.plot)
            plt.savefig('figures/exact_min_psd.pdf', dpi=300, bbox_inches='tight')

            # max ev, psd matrix, psd input
            exact_curves, _ = EDExperiments(dim_z=5, method='exact', loss_on='max', mat_type='psd', rand_in=False)

            plt.figure()
            PlotResults(exact_curves, fcn=plt.plot)
            plt.savefig('figures/exact_max_psd.pdf', dpi=300, bbox_inches='tight')
        
        if pytorch_exp:
            # all evs, psd matrix, random input
            exact_curves, _ = EDExperiments(dim_z=5, method='pytorch', loss_on='all', mat_type='psd', rand_in=True)

            plt.figure()
            PlotResults(exact_curves, fcn=plt.plot)
            plt.savefig('figures/pytorch_all_psd_rand.pdf', dpi=300, bbox_inches='tight')
            
            # second smallest ev, psd matrix, random input
            exact_curves, _ = EDExperiments(dim_z=5, method='pytorch', loss_on='second_smallest', mat_type='psd', rand_in=True)

            plt.figure()
            PlotResults(exact_curves, fcn=plt.plot)
            plt.savefig('figures/pytorch_second_psd_rand.pdf', dpi=300, bbox_inches='tight')

            # min ev, psd matrix, random input
            exact_curves, _ = EDExperiments(dim_z=5, method='pytorch', loss_on='min', mat_type='psd', rand_in=True)

            plt.figure()
            PlotResults(exact_curves, fcn=plt.plot)
            plt.savefig('figures/pytorch_min_psd_rand.pdf', dpi=300, bbox_inches='tight')

            # max ev, psd matrix, random input
            exact_curves, _ = EDExperiments(dim_z=5, method='pytorch', loss_on='max', mat_type='psd', rand_in=True)

            plt.figure()
            PlotResults(exact_curves, fcn=plt.plot)
            plt.savefig('figures/pytorch_max_psd_rand.pdf', dpi=300, bbox_inches='tight')
            
            # all evs, psd matrix, psd input
            exact_curves, _ = EDExperiments(dim_z=5, method='pytorch', loss_on='all', mat_type='psd', rand_in=False)

            plt.figure()
            PlotResults(exact_curves, fcn=plt.plot)
            plt.savefig('figures/pytorch_all_psd.pdf', dpi=300, bbox_inches='tight')
            
            # second smallest ev, psd matrix, psd input
            exact_curves, _ = EDExperiments(dim_z=5, method='pytorch', loss_on='second_smallest', mat_type='psd', rand_in=False)

            plt.figure()
            PlotResults(exact_curves, fcn=plt.plot)
            plt.savefig('figures/pytorch_second_psd.pdf', dpi=300, bbox_inches='tight')

            # min ev, psd matrix, psd input
            exact_curves, _ = EDExperiments(dim_z=5, method='pytorch', loss_on='min', mat_type='psd', rand_in=False)

            plt.figure()
            PlotResults(exact_curves, fcn=plt.plot)
            plt.savefig('figures/pytorch_min_psd.pdf', dpi=300, bbox_inches='tight')

            # max ev, psd matrix, psd input
            exact_curves, _ = EDExperiments(dim_z=5, method='pytorch', loss_on='max', mat_type='psd', rand_in=False)

            plt.figure()
            PlotResults(exact_curves, fcn=plt.plot)
            plt.savefig('figures/pytorch_max_psd.pdf', dpi=300, bbox_inches='tight')

    # if enable_ed_random_exp:
    #     plt.show()
