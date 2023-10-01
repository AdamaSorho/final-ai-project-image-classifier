from torch import optim

def get_optimizer(model, lr):
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    
    return optimizer
