import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from timm import create_model  # PyTorch image models library
import csv
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Loads the data
def data_loader(directory, batch_size):
    # Data Preprocessing
    transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dir = directory + "\\train"
    val_dir = directory + "\\val"
    val2_dir = directory + "\\test"

    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)
    val2_dataset = datasets.ImageFolder(root=val2_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    val2_loader = DataLoader(val2_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, val2_loader


# Vision Transformer Model
class ViTClassifier(nn.Module):
    def __init__(self, hidden_classes=3, num_classes=2):
        super(ViTClassifier, self).__init__()
        self.model = create_model('vit_base_patch16_224', pretrained=True)  # loads a pre-trained ViT model
        self.model.head = nn.Linear(self.model.head.in_features,
                                    hidden_classes)  # customize the model head to have 3 classes for the output (fire, smoke, non-fire)
        self.final = nn.Linear(hidden_classes,
                               num_classes)  # 2 neurons because it's the number of actual outputs that we want

    def forward(self, x):
        return self.final(self.model(x))


# 4. Training Loop
def train_epoch(model, loader, optimizer, criterion, device):
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

        i += batch_size
        print("Running average train loss per image: " + str(running_loss / i))
        break

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


# Validation Loop
def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    accuracy = correct / total
    return epoch_loss, accuracy


# Main
directory = "dl2425_challenge_dataset"
batch_size = 4
train_loader, val_loader, val2_loader = data_loader(directory, batch_size)

model = ViTClassifier(hidden_classes=3, num_classes=2)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
model.to(device)

epochs = 4
dir_models = "saved_models\\"
model_name = "VitModel"
LOAD_MODEL = True # set to True if you want to load a model to continue its training
START_EPOCH = 0  # the epoch from which you want to continue the training. Leave to 0 if you want to train from scratch

# Create and write the CSV header before starting the training loop
csv_file = f"{model_name}_metrics.csv"
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Train Loss", "Val Loss", "Val Accuracy", "Test Loss", "Test Accuracy"])  # Write header

for epoch in range(epochs):

    if LOAD_MODEL:
        model.load_state_dict(torch.load(dir_models + model_name + "_epoch_" + str(START_EPOCH) + ".pt",
                                         weights_only=True))  # loads the correct weights in the model
        model.eval()

    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

    torch.save(model.state_dict(), dir_models + model_name + "_epoch_" + str(START_EPOCH + epoch + 1) + ".pt")
    model.state_dict()

    val_loss, val_accuracy = validate_epoch(model, val_loader, criterion, device)
    val2_loss, val2_accuracy = validate_epoch(model, val2_loader, criterion, device)

    # Write the current epoch's metrics to the CSV file
    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([epoch+1, train_loss, val_loss, val_accuracy, val2_loss, val2_accuracy])

    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Test Loss: {0:.4f}, "
          f"Test Accuracy: {0:}")


# Graphs for visualization
import pandas as pd

# Read the CSV file
df = pd.read_csv(csv_file)

# Plot Train Loss and Val Loss
plt.figure(figsize=(10, 6))
plt.plot(df["Epoch"], df["Train Loss"], label="Train Loss", marker='o')
plt.plot(df["Epoch"], df["Val Loss"], label="Val Loss", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train and Validation Loss")
plt.legend()
plt.grid()
plt.savefig(f"{model_name}_TrainVal_Loss.png")  # Save the plot as an image
plt.show()

# Plot Val Accuracy and Test Accuracy
plt.figure(figsize=(10, 6))
plt.plot(df["Epoch"], df["Val Accuracy"], label="Val Accuracy", marker='o')
plt.plot(df["Epoch"], df["Test Accuracy"], label="Test Accuracy", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Validation and Test Accuracy")
plt.legend()
plt.grid()
plt.savefig(f"{model_name}_ValTest_Accuracy.png")  # Save the plot as an image
plt.show()