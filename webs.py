from flask import Flask, request, render_template
from opencv import cv2
import torch
import torchvision.transforms as transforms
from anewhope import CustomCNN

app = Flask(__name__)

# Load the trained model
model = torch.load('anewhope.pth', map_location=torch.device('cpu'))

# Define a function to make predictions from camera input
def predict_from_camera():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Capture a frame from the webcam
    ret, frame = cap.read()

    # Preprocess the frame for model input
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_image = transform(frame).unsqueeze(0)

    # Get the model prediction
    with torch.no_grad():
        outputs = model(input_image)
        _, predicted = torch.max(outputs.data, 1)
        prediction_label = labels[predicted.item()]

    # Release the webcam
    cap.release()

    return prediction_label

# Define a route to handle prediction requests
@app.route('/predict', methods=['GET'])
def predict_api():
    # Make a prediction from camera input
    prediction = predict_from_camera()

    # Return the prediction
    return jsonify({'prediction': prediction})

# Define a route to render the main web page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Start the web server
if __name__ == '__main__':
    app.run(debug=True)
