import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from timm import create_model # PyTorch image models library

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
    train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# Vision Transformer Model
class ViTClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(ViTClassifier, self).__init__()
        self.model = create_model('vit_base_patch16_224', pretrained=True) # loads a pre-trained ViT model
        self.model.head = nn.Linear(self.model.head.in_features, num_classes) # customize the model head to have 3 classes for the output

    def forward(self, x):
        return self.model(x)

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
        print("Running average train loss per image: " + str(running_loss/i))

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
train_loader, val_loader = data_loader(directory, batch_size)

model = ViTClassifier(num_classes=3)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
model.to(device)

epochs = 1
for epoch in range(epochs):

    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_accuracy = validate_epoch(model, val_loader, criterion, device)
    
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, "
          f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")