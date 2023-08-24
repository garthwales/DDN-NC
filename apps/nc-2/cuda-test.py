import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # just use one GPU on big machine
import torch
assert torch.cuda.device_count() == 1