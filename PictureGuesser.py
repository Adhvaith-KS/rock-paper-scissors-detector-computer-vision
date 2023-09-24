import torch
from torchvision.transforms import transforms
from PIL import Image
from anewhope import CustomCNN  # Importing my trained model class, which has to have the same name as the .pth file

# Loading my trained model
model = CustomCNN()
model.load_state_dict(torch.load('anewhope.pth', map_location=torch.device('cpu')))
model.eval()

# Defining all labels for rock, paper, scissors, and other
labels = ['Other', 'Paper', 'Rock', 'Scissor']

# Loading and preprocessing the image
image_path = input("Enter the path to image: ")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
image = Image.open(image_path)
input_image = transform(image).unsqueeze(0)

# Getting a prediction
with torch.no_grad():
    outputs = model(input_image)
    _, predicted = torch.max(outputs.data, 1)
    prediction_label = labels[predicted.item()]

print(f'The image is classified as: {prediction_label}')
