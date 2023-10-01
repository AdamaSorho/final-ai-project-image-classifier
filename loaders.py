from torchvision import transforms, datasets
import torch

def get_trainloader(data_directory):
    train_dir = data_directory + '/train'
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])])
    
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    
    # map classes to indices
    class_to_idx = train_data.class_to_idx
    
    return trainloader, class_to_idx

def get_validloader(data_directory):
    valid_dir = data_directory + '/valid'
    eval_transforms = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                 [0.229, 0.224, 0.225])])
    eval_data = datasets.ImageFolder(valid_dir, transform=eval_transforms)
    evalloader = torch.utils.data.DataLoader(eval_data, batch_size=64)
    
    return evalloader