import torch
from torch import nn

import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torchvision import transforms

from matplotlib import pyplot as plt

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential( # the output of one layer is the input of the next one
            nn.Flatten(), # converts the input into a linear vector
            nn.Linear(32 * 32 * 3, 64), # takes the input, multiplies it by the weight and adds the bias. Here the input is 32*32*3 and the output 64
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 10) # we'll have 10 classes
        )

    def forward(self, x):
        return self.layers(x) # passes x into the layers
   
def train(PATH, load_model):
    torch.manual_seed(42) # Set fixed random number seed

    # Prepare CIFAR-10 dataset
    dataset = CIFAR10(PATH, download=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(dataset, train = True, batch_size=10, shuffle=True, num_workers=1)

    mlp = MLP() # Initialize the MLP
    if load_model: # to continue from a checkpoint
        mlp.load_state_dict(torch.load(PATH + "mlp_model.pt", weights_only=True)) # loads the correct weights in the model
        mlp.eval()


    # Define the loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4) # decides how to adjust the parameters after the backward pass
  
    # Run the training loop
    for epoch in range(0, 5): # 5 epochs at maximum
        print(f'Starting epoch {epoch+1}')
        current_loss = 0.0 # Set current loss value
        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):
            inputs, targets = data
            optimizer.zero_grad() # sets the gradient to 0. It's important when we change batch so to not have the previous gradients
            outputs = mlp(inputs) # Perform forward pass
            loss = loss_function(outputs, targets) # Compute loss
            loss.backward() # Perform backward pass
            optimizer.step() # Perform optimization
            # Print statistics
            current_loss += loss.item()
            if i % 500 == 499:
                print('Loss after mini-batch %5d: %.3f' %
                    (i + 1, current_loss / 500))
                current_loss = 0.0

    # Process is complete.
    print('Training process has finished.')

    torch.save(mlp.state_dict(), PATH + "mlp_model.pt")
    mlp.state_dict()

def load(PATH):
    mlp = MLP() # initializes an empty model
    mlp.load_state_dict(torch.load(PATH + "mlp_model.pt", weights_only=True)) # loads the correct weights in the model
    mlp.eval()

    dataset = CIFAR10(PATH, train = True, download=False, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)
    loss_function = nn.CrossEntropyLoss()
    
    current_loss = 0
    for i, data in enumerate(trainloader, 0):
        inputs, targets = data
        outputs = mlp(inputs) # Perform forward pass
        loss = loss_function(outputs, targets) # Compute loss
        # Print statistics
        current_loss += loss.item()
    
    print('Loss : ', current_loss / len(trainloader)) # as many as the elements in the mini-batch * batch_size


def test(PATH):
    # Load the model
    mlp = MLP()
    mlp.load_state_dict(torch.load(PATH + "mlp_model.pt", weights_only=True))
    mlp.eval()

    # Load the dataset
    dataset = CIFAR10(PATH, train = False, download=False, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False, num_workers=1)
    loss_function = nn.CrossEntropyLoss()

    # Compute loss for the entire test dataset
    current_loss = 0
    for i, data in enumerate(testloader, 0):
        inputs, targets = data
        outputs = mlp(inputs) # Perform forward pass
        loss = loss_function(outputs, targets) # Compute loss
        current_loss += loss.item()

    avg_loss = current_loss / len(testloader) # Calculate the average loss
    print('Average Loss on test set: ', avg_loss)


def show_output_one_image(PATH, image_index, columns, rows):
    # Load the model
    mlp = MLP()
    mlp.load_state_dict(torch.load(PATH + "mlp_model.pt", weights_only=True))
    mlp.eval()

    # Load the dataset
    dataset = CIFAR10(PATH, train = False, download=False, transform=transforms.ToTensor())
    
    fig = plt.figure(figsize=(8, 8)) # sets the size of each image
    for i in range(1, columns*rows + 1):
        image, target_class = dataset[image_index] # Load the specified image and target
        image_for_output = image.unsqueeze(0) # Add batch dimension
        output = mlp(image_for_output) # Get model output
        predicted_class = output.argmax(1).item() # Predicted class index
        class_names = dataset.classes # takes the whole list of class names
        image = image.permute(1, 2, 0).numpy() #swaps the order of channels to match RGB
        fig.add_subplot(rows, columns, i) # adds a subplot to show that image
        plt.title("Target class: " + class_names[target_class] + "\n Predict class: " + class_names[predicted_class]) # the title
        plt.axis('off')  # Turn off axis labels
        plt.imshow(image) # displays the image
        image_index += 1 # increases the counter to show the next image
    
    plt.show()
    # print(f"Predicted Class: {predicted_class}, Target Class: {target_class}") # in alternative you can print the result






if __name__ == '__main__':
    PATH = "C:\\Users\\gabri\\Documents\\Davide\\Notes\\Python\\Machine Learning\\data\\"
    #train(PATH, True)
    #load(PATH)
    #test(PATH)
    #show_output_one_image(PATH, 2, 5, 2)