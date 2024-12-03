import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from timm import create_model  # PyTorch image models library
import csv
import matplotlib.pyplot as plt
import pandas as pd
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Loads and normalizes the data and returns the data loaders
def DataLoader(directory, batch_size):
    # Data Preprocessing
    transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dir = directory + "\\train"
    val_dir = directory + "\\val"
    test_dir = directory + "\\test"

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


# Vision Transformer Model
class ViTClassifier(nn.Module):
    def __init__(self, hidden_classes=3, num_classes=2):
        super(ViTClassifier, self).__init__()
        self.model = create_model('vit_base_patch16_224', pretrained=True)  # loads a pre-trained ViT model with 16x16 patches
        self.model.head = nn.Linear(self.model.head.in_features,
                                    hidden_classes)  # customize the model head to have 3 classes for the output (fire, smoke, non-fire)
        self.final = nn.Linear(hidden_classes,
                               num_classes)  # 2 neurons because it's the number of actual outputs that we want

    def forward(self, x):
        return self.final(self.model(x))


# Training Loop
def TrainEpoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    i = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        batch_size = images.size(0)
        running_loss += loss.item() * batch_size

        i += batch_size # used to find the loss per batch
        print("Running average train loss per image: " + str(running_loss / i))

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


# Validation Loop
def ValidateEpoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for j, (images, labels) in enumerate(loader, 0):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            # does the predictions and sums the correct ones
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if preds != labels:
                # returns the name of the image classified wrongly
                print("actual label: " + str(labels) + ", pred: " + str(preds) + ", image: " + loader.dataset.samples[j][0])
            
            
    epoch_loss = running_loss / len(loader.dataset)
    accuracy = correct / total
    return epoch_loss, accuracy


# Plot graphs
def PlotGraphs(csv_file, model_name):
    # Graphs for visualization

    # Read the CSV file
    df = pd.read_csv(csv_file, sep =",")
    plt.rcParams.update({'font.size': 18}) # updates the font size in all the plots
    # Plot Train Loss and Val Loss
    plt.figure(figsize=(10, 6))
    plt.plot(df["Epoch"], df["Train Loss"], label="Train Loss", marker='o')
    plt.plot(df["Epoch"], df["Val Loss"], label="Val Loss", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(model_name + "Train and Validation Loss")
    plt.legend()
    plt.grid()
    plt.savefig(model_name + "_TrainVal_Loss.png")  # Save the plot as an image
    plt.show()

    # Plot Val Accuracy and Test Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(df["Epoch"], df["Val Accuracy"], label="Val Accuracy", marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title(model_name + "Validation Accuracy")
    plt.legend()
    plt.grid()
    plt.savefig(model_name + "_ValTest_Accuracy.png")  # Save the plot as an image
    plt.show()


# Performs both training and validation and saves the losses
def TrainAndValidate(model, train_loader, val_loader, model_metrics, epochs):
    
    # Create and write the CSV header before starting the training loop
    file = open(model_metrics, mode="w", newline="")
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Train Loss", "Val Loss", "Val Accuracy"])  # Write header
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    
    for epoch in range(epochs):
        train_loss = TrainEpoch(model, train_loader, optimizer, criterion, device)
        torch.save(model.state_dict(), dir_models + model_name + "_epoch_" + str(START_EPOCH + epoch + 1) + ".pt")
        model.state_dict()
        
        val_loss, val_accuracy = ValidateEpoch(model, val_loader, criterion, device)
        
        # Write the current epoch's metrics to the CSV file
        writer.writerow([epoch+1, train_loss, val_loss, val_accuracy])
        
        print("Epoch " + str(epoch + 1) + "/" + str(epochs) + ", Train Loss: " + str(round(train_loss, 4)) + "\n" +
              "Val Loss: " + str(round(val_loss, 4)) + ", Val Accuracy: " + str(round(val_accuracy, 4)))
    
    file.close()

# Runs the model, either with training and validation or printing the results
def RunModel(directory, batch_size, epochs, dir_models, model_name, LOAD_MODEL, START_EPOCH):
    train_loader, val_loader, test_loader = DataLoader(directory, batch_size)
    model = ViTClassifier(hidden_classes=3, num_classes=2)
    model.to(device)

    model_metrics = model_name + "_metrics.csv"
    test_results = "test_results.csv"

    # This will run only if we want to load the weights of our model instead of starting from zero
    if LOAD_MODEL:
        model.load_state_dict(torch.load(dir_models + model_name + "_epoch_" + str(START_EPOCH) + ".pt",
                                        weights_only=True, map_location=torch.device('cpu')))  # loads the correct weights in the model
        model.eval()
    
    TrainAndValidate(model, train_loader, val_loader, model_metrics, epochs)
    PlotGraphs(model_metrics, model_name)

directory = "dl2425_challenge_dataset"
batch_size = 4
epochs = 4
dir_models = "saved_models\\"
model_name = "VitModel"
LOAD_MODEL = False # set to True if you want to load a model to continue its training
START_EPOCH = 1 # the epoch from which you want to continue the training. Leave to 0 if you want to train from scratch


RunModel(directory, batch_size, epochs, dir_models, model_name, LOAD_MODEL, START_EPOCH)
