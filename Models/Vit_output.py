import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from timm import create_model  # PyTorch image models library
import matplotlib.pyplot as plt
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loads and normalizes the data and returns the data loader
def DataLoader(test_folder, batch_size):
    # Data Preprocessing
    transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    test_dataset = datasets.ImageFolder(root=test_folder, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

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

# Returns the csv file for the test images
def TestOutput(model, loader, device, out_file):
    model.eval()
    f = open(out_file, 'w')
    f.write('id,class\n')
    with torch.no_grad():
        for j, (images, _) in enumerate(loader, 0):
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1) # does the prediction
            f.write(loader.dataset.samples[j][0] + ',' + str(preds.item()) + '\n')
            f.flush()
    f.close()

# Main
test_folder = "dl2425_challenge_dataset"
batch_size = 1
test_loader = DataLoader(test_folder, batch_size)

model = ViTClassifier(hidden_classes=3, num_classes=2)
model.to(device)

dir_models = "saved_models\\"
model_name = "VitModel"
model_epoch = 2  # the last epoch of the model you want to train

model.load_state_dict(torch.load(dir_models + model_name + "_epoch_" + str(model_epoch) + ".pt",
                                 weights_only=True, map_location=torch.device('cpu')))  # loads the correct weights in the model

out_file = "out.txt"

TestOutput(model, test_loader, device, out_file)
