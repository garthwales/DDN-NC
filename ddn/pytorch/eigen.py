import torch
import cv2
import numpy as np
from ddn.pytorch.node import AbstractDeclarativeNode


# TODO: implement multiple things in here
class NC(AbstractDeclarativeNode):
    def __init__(self,
        eps=1e-8,
        gamma=None,
        chunk_size=None,
        objective_type='v1'): # obj type to test different objective functions...
        super().__init__(eps=eps, gamma=gamma, chunk_size=chunk_size)
        self.objective_type = objective_type
        
    def pre_solve(self, *xs):
        return 
        
    def objective(self, *xs, y):
        return super().objective(*xs, y=y)
    
    def solve(self, *xs):
        if self.objective_type == 'v1':
            return
        elif self.objective_type == 'v2':
            return
        
        
        return super().solve(*xs)
    
    
    
    
    # simple network
    # reshape to mxm
    # relu
    
    # x = torch.matmul(x, x.transpose(1, 2)) # positive definite :)
        # or
    # x = torch.matmul(u[:, :, 0:n], u[:, :, 0:n].transpose(1, 2)) # for n+1 rank matrix instead
    
    # make into deterministic form.. otherwise it doesn't matter if it works or not if can flip flop
    # e.g. use 
    # def uniform_solution_direction(self, u, u_ref=None):
        # batch, m, n = u.shape
        # direction_factor = 1.0

        # if self.uniform_solution_method != 'skip':
        #     if u_ref is None:
        #         u_ref = u.new_ones(1, m, 1).detach()

        #     direction = torch.einsum('bmk,bmn->bkn', u_ref, u)

        #     if u_ref.shape[2] == n:
        #         direction = torch.diagonal(direction, dim1=1, dim2=2).view(batch, 1, n)

        #     if self.uniform_solution_method == 'positive':
        #         direction_factor = (direction >= 0).float()
        #     elif self.uniform_solution_method == 'negative':
        #         direction_factor = (direction <= 0).float()

        # u = u * (direction_factor - 0.5) * 2

        # return u 