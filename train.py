import torch
from torch import nn, optim
from torchvision import models

from get_input_args import get_input_args
from loaders import get_trainloader, get_validloader
from classifier import get_classifier
from available_device import get_device
from save_model import save_model

def main():
    args = get_input_args()
    trainloader, class_to_idx = get_trainloader(args.data_directory[0])
    evalloader = get_validloader(args.data_directory[0])

    model = getattr(models, args.arch)(pretrained=True)

    # Freeze parameters so we don't backprob through them
    for param in model.parameters():
        param.requires_grad = False

    classifier = get_classifier(args.hidden_units, args.arch)
    model.classifier = classifier
    device = get_device(args.gpu)
    model.to(device)

    criterion = nn.NLLLoss()
    # Only train the classifier parameters. Features parameters are frozen.
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    epochs = args.epochs
    steps = 0
    running_loss = 0
    print_every = 5

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps +=1

            # move inputs, labels to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            # reset the grad
            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                eval_loss = 0
                accuracy = 0
                model.eval()

                with torch.no_grad():
                    for inputs, labels in evalloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        eval_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"validation loss: {eval_loss/len(evalloader):.3f}.. "
                  f"validation accuracy: {accuracy/len(evalloader):.3f}")

                running_loss = 0
                model.train()    
    
    # save model
    save_model(model, classifier, optimizer, criterion, 
               class_to_idx, epochs, args.save_dir, args.arch)

# Call to main function to run the program
if __name__ == "__main__":
    main()
