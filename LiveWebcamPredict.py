import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from modelclass import CustomCNN  # Importing my trained model class, which has to have the same name as the .pth file
import numpy as np

# Loading my trained model
model = CustomCNN(num_classes=4)  
model.load_state_dict(torch.load('anewhope.pth', map_location=torch.device('cpu')))  
model.eval()

# Defining the labels for rock, paper, scissors, and other
labels = ['Other', 'Paper', 'Rock', 'Scissor']

# Initializing webcam
cap = cv2.VideoCapture(0)

while True:
    # Capturing a frame from the webcam
    ret, frame = cap.read()

    # Converting the frame to a PIL image
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Preprocessing the PIL image for model input
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resizing to match the model's input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_image = transform(pil_image).unsqueeze(0)

    # Getting model prediction
    with torch.no_grad():
        outputs = model(input_image)
        _, predicted = torch.max(outputs.data, 1)
        prediction_label = labels[predicted.item()]

    # Displaying the frame with the prediction
    cv2.putText(frame, f'Gesture: {prediction_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Rock Paper Scissors Detection', frame)

    # Allowing anyone to press 'q' to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Closing all OpenCV windows
cap.release()
cv2.destroyAllWindows()
