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