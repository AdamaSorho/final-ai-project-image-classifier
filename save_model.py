import torch

def save_model(model, classifier, optimizer, criterion, class_to_idx, epochs, directory, arch):
    checkpoint = {'state_dict': model.state_dict(),
                 'classifier': classifier,
                 'optimizer': optimizer, 
                 'criterion': criterion,
                  'arch': arch,
                  'class_to_idx': class_to_idx,
                 'epochs': epochs}
    
    torch.save(checkpoint, directory)
