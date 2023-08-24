# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" # just use one GPU on big machine
import torch
# assert torch.cuda.device_count() == 1
print(torch.cuda.device_count())

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(1))
    print('Memory Usage:')
    print(torch.cuda.memory_summary(0))
    
    A = torch.randn((96*96, 96*96), device=device)
    
    print(torch.cuda.get_device_name(1))
    print('Memory Usage:')
    print(torch.cuda.memory_summary(0))
    print(torch.cuda.memory_summary(1))