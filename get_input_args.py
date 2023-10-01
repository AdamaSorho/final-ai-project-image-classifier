import argparse

def get_input_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('data_directory', nargs=1)
    parser.add_argument('--save_dir', default="checkpoint.pth")
    parser.add_argument('--arch', default='vgg11', type=str, choices=['vgg11', 'vgg13', 
                                                                      'vgg16', 'alexnet', 'densenet121'])
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--hidden_units', default=3096, type=int)
    parser.add_argument('--epochs', default=2, type=int)
    parser.add_argument('--gpu', action='store_true')

    args = parser.parse_args()
    
    return args
