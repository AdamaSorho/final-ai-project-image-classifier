from torch import nn
from collections import OrderedDict

def get_classifier(hidden_units, arch):
    if 'densenet121' in arch:
        n_input = 1024
    elif 'alexnet' in arch:
        n_input = 9216
    else:
       n_input = 25088 
       
    classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(n_input, hidden_units)),
                                       ('relu1', nn.ReLU(inplace=True)),
                                       ('dropout1', nn.Dropout()),
                                       ('fc2', nn.Linear(hidden_units, hidden_units)),
                                       ('relu2', nn.ReLU(inplace=True)),
                                       ('dropout2', nn.Dropout()),
                                       ('fc3', nn.Linear(hidden_units, 102)),
                                       ('output', nn.LogSoftmax(dim=1))]))

    return classifier
