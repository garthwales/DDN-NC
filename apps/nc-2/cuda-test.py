# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" # just use one GPU on big machine
import torch

import os
 
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# assert torch.cuda.device_count() == 1
print('device count')
print(torch.cuda.device_count())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

print('current device')
print(torch.cuda.current_device())

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(1))
    print('Memory Usage:')
    print(torch.cuda.memory_allocated())
        
    A = torch.randn((96*96, 96*96), device=device)
    
    print('current device')
    print(torch.cuda.current_device())
    
    print('Memory Usage:')
    print(torch.cuda.memory_allocated())