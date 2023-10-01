import torch
from torchvision import models

def load_checkpoint(filepath):
    filepath += '.pth'
    checkpoint = torch.load(filepath)
    model = getattr(models, checkpoint['arch'])()
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    # Freeze parameters so we don't backprob through them
    for param in model.parameters():
        param.requires_grad = False

    return model
