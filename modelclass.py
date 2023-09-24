#this file's name has to be the same as the model (.pth) file's name
import torch
import torch.nn as nn
import torchvision.models as models

class CustomCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(CustomCNN, self).__init__()

        # Using a pretrained ResNet-18 model as the backbone
        self.backbone = models.resnet18(pretrained=False)  # Setting pretrained to False because I have my own model trained on my own dataset
        # Replacing the final classification layer for the desired number of classes
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.backbone(x)
