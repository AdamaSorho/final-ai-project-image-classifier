import torch

def get_device(is_gpu):
    if is_gpu and torch.cuda.is_available():
        return 'cuda:0'
    elif is_gpu and torch.backends.mps.is_available():
        return 'mps'
    
    return 'cpu'
        
    
