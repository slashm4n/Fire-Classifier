import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define hyperparameters
batch_size = 32
learning_rate = 0.001
num_epochs = 20
import cv2

# Path to one sample image
sample_image_path = "dl2425_challenge_dataset/train/0/0.jpg"

# Load the image using OpenCV
image = cv2.imread(sample_image_path)

# Get the dimensions
height, width, channels = image.shape
input_size = height * width * channels  # Assuming images are 64x64 with 3 channels
num_classes = 3  # No fire/smoke, smoke, fire

# Define data transformations
transform = transforms.ToTensor()

# Load datasets
train_dataset = datasets.ImageFolder(root="dl2425_challenge_dataset/train", transform=transform)
val_dataset = datasets.ImageFolder(root="dl2425_challenge_dataset/val", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define the MLP model
class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),  # Flatten the input images
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        return self.model(x)

# Instantiate the model
hidden_size = 512
model = MLPClassifier(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
def train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer):
    best_val_accuracy = 0.0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_accuracy = 100 * correct / total
        val_accuracy = evaluate_model(model, val_loader)

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Loss: {train_loss/len(train_loader):.4f}, "
              f"Train Accuracy: {train_accuracy:.2f}%, "
              f"Val Accuracy: {val_accuracy:.2f}%")

        # Save the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), "best_model.pth")

    print("Training complete. Best Validation Accuracy: {:.2f}%".format(best_val_accuracy))

# Evaluation function
def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    return accuracy

# Train the model
train_model(model, train_loader, val_loader, num_epochs, criterion, optimizer)

# Test the model
def predict_on_test(model, test_dir):
    test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    predictions = []

    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())

    return predictions

# Predict on the test set
#test_dir = "dl2425_challenge_dataset/test"
#test_predictions = predict_on_test(model, test_dir)

'''
# Save predictions to a file
with open("test_predictions.txt", "w") as f:
    for pred in test_predictions:
        f.write(f"{pred}\n")

'''
