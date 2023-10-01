import torch

from predict_input_args import get_input_args
from available_device import get_device
from process_image import process_image
from load_category_names import get_names
from load_checkpoint import load_checkpoint

def main():
    args = get_input_args()
    device = get_device(args.gpu)
    model = load_checkpoint(args.checkpoint[0])
    
    probs, classes = predict(args.image_directory[0], model, device, args.top_k)
    
    category_names = get_names(args.category_names)
    
    # Convert probs and classes to list in case top_k is 1
    if isinstance(probs, float) or isinstance(classes, int):
        probs = [probs]
        classes = [classes]
        
    for index, prob in enumerate(probs):
        class_name = category_names[str(classes[index])]
        print(f'Flower name: {class_name}, probability: {prob}')
    

def predict(image_path, model, device, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Process the image
    image = process_image(image_path)
    
    # Add a batch dimension to the image
    image = image.unsqueeze(0)
    
    model.to(device)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Disable gradient calculation
    with torch.no_grad():
        # use the device available
        image = image.to(device)
        # Forward pass through the model
        output = model.forward(image)
        
        # Calculate the probabilities
        probabilities = torch.exp(output)
        
        # Get the top k probabilities and classes
        top_probabilities, top_indices = probabilities.topk(topk, dim=1)
        
        
    # Convert the tensors to lists
    top_probabilities = top_probabilities.squeeze().tolist()
    top_indices = top_indices.squeeze().tolist()
    
    # get indices classes
    idx_to_classes = {idx: cls for cls, idx in model.class_to_idx.items()}
    top_classes = [idx_to_classes[idx] for idx in top_indices]
    
    return top_probabilities, top_classes
    
# Call to main function to run the program
if __name__ == "__main__":
    main()
    
    
