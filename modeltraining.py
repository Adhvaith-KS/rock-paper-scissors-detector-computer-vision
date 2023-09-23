import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torch.optim import lr_scheduler
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from sklearn.utils import class_weight

# Setting the device to GPU because I have a gaming laptop, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Defining my custom model architecture by creating a class that inherits from nn.Module.
class CustomCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(CustomCNN, self).__init__()

        # Usimg a pretrained ResNet-18 model as the backbone
        self.backbone = models.resnet18(pretrained=True)
        # Replacing the final classification layer for the desired number of classes
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.backbone(x)

# Defining all my data transformations including data augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(degrees=30),  # Randomly rotate images by up to 30 degrees
    transforms.RandomVerticalFlip(),  # Randomly flip images vertically
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Randomly translate images
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),  # Random perspective transformations
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(256),  # Resizing the image to 256x256 pixels
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Randomly adjust brightness, contrast, and saturation
    transforms.RandomRotation(degrees=30),  # Randomly rotate the image by up to 30 degrees
    transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
    transforms.ToTensor(),  # Convert the image to a PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize the image
])


# Specify the dataset location
dataset_root = 'D:\\VScode\\A new hope\\image_data'  # Replace with the path to your dataset folder (yes I name my projet folders after the names of star wars movies)

# Loading the dataset with data augmentation
dataset = datasets.ImageFolder(root=dataset_root, transform=train_transform)

# Calculated class weights to address class imbalance
class_weights = class_weight.compute_class_weight('balanced', classes=[0, 1, 2, 3], y=dataset.targets)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

# Splitting the dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Creating data loaders with weighted sampling
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, sampler=None)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Instantiating the model to move it to the GPU
model = CustomCNN(num_classes=4).to(device)

# Loss function and optimizer with class weights
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)  # Learning rate scheduler

# Training loop with early stopping
num_epochs = 30  # Increased my number of epochs from 20 to 30, good decision
best_val_accuracy = 0.0
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    scheduler.step()
    
    # Print training loss (optional)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    # Evaluated the model on the validation set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_accuracy = 100 * correct / total
    print(f'Validation Accuracy: {val_accuracy:.2f}%')
    
    # Saving the model if it achieves the best validation accuracy
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), 'anewhope.pth')
    
    # Early stopping if the loss is below 0.4
    if loss.item() < 0.4:
        print("Lite")
        break

print(f'Best Validation Accuracy: {best_val_accuracy:.2f}%')
