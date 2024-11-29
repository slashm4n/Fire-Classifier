import torch
from torch import nn

import torch
from torch import nn
from torchvision.datasets import CIFAR10
from torchvision import transforms

from matplotlib import pyplot as plt
class CNN(nn.Module):
#  Determine what layers and their order in CNN object 
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv_layer2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.conv_layer3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.conv_layer4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        
        self.fc1 = nn.Linear(1600, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
    
    # Progresses data across layers    
    def forward(self, x):
        out = self.conv_layer1(x)
        out = self.conv_layer2(out)
        out = self.max_pool1(out)
        
        out = self.conv_layer3(out)
        out = self.conv_layer4(out)
        out = self.max_pool2(out)
                
        out = out.reshape(out.size(0), -1)
        
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out




def train(PATH, num_classes, load_model):
    torch.manual_seed(42) # Set fixed random number seed

    # Prepare CIFAR-10 dataset
    all_transforms = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(),transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
    dataset = CIFAR10(PATH, train=True, download=False, transform = all_transforms)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)
    total_step = len(trainloader)
    cnn = CNN(num_classes) # Initialize the CNN
    if load_model: # to continue from a checkpoint
        cnn.load_state_dict(torch.load(PATH + "cnn_model.pt", weights_only=True)) # loads the correct weights in the model
        cnn.eval()


    # Define the loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(cnn.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)   # decides how to adjust the parameters after the backward pass
  
    # Run the training loop
    for epoch in range(5):
    # Load in the data in batches using the train_loader object
        for i, (images, labels) in enumerate(trainloader):  
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = cnn(images)
            loss = loss_function(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))  

    torch.save(cnn.state_dict(), PATH + "cnn_model.pt")
    cnn.state_dict()

def load(PATH, num_classes):
    cnn = CNN(num_classes) # initializes an empty model
    cnn.load_state_dict(torch.load(PATH + "cnn_model.pt", weights_only=True)) # loads the correct weights in the model
    cnn.eval()
    
    all_transforms = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(),transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
    dataset = CIFAR10(PATH, train = True, download=False, transform = all_transforms)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)
    loss_function = nn.CrossEntropyLoss()
    
    for epoch in range(5):
        # Load in the data in batches using the train_loader object
            for i, (images, labels) in enumerate(trainloader):  
                # Move tensors to the configured device
                images = images.to(device)
                labels = labels.to(device)
                # Forward pass
                outputs = cnn(images)
                loss = loss_function(outputs, labels)

            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))  
    
    
def test(PATH, num_classes):
    # Load the model
    cnn = CNN(num_classes)
    cnn.load_state_dict(torch.load(PATH + "cnn_model.pt", weights_only=True))
    cnn.eval()

    # Load the dataset
    all_transforms = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(),transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
    dataset = CIFAR10(PATH, train = False, download=False, transform = all_transforms)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=False, num_workers=1)
    loss_function = nn.CrossEntropyLoss()

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = cnn(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        print('Accuracy of the network on the {} train images: {} %'.format(50000, 100 * correct / total))


def show_output_one_image(PATH, num_classes, image_index, columns, rows):
    # Load the model
    cnn = CNN(num_classes)
    cnn.load_state_dict(torch.load(PATH + "cnn_model.pt", weights_only=True))
    cnn.eval()

    # Load the dataset
    all_transforms = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor(),transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])])
    dataset = CIFAR10(PATH, train = False, download=False, transform = all_transforms)
    
    fig = plt.figure(figsize=(8, 8)) # sets the size of each image
    for i in range(1, columns*rows + 1):
        image, target_class = dataset[image_index] # Load the specified image and target
        image_for_output = image.unsqueeze(0) # Add batch dimension
        output = cnn(image_for_output) # Get model output
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
    batch_size = 64
    num_classes = 10
    learning_rate = 0.001
    num_epochs = 20
    
    # Use transforms.compose method to reformat images for modeling,
    # and save to variable all_transforms for later use


    # Device will determine whether to run the training on GPU or CPU.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #train(PATH, num_classes, False)
    #load(PATH, num_classes)
    #test(PATH, num_classes)
    show_output_one_image(PATH, num_classes, 2, 2, 1)