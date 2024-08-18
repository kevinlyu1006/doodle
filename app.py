from flask import Flask, render_template, request, jsonify
# from your_model_file import predict_doodle
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


app = Flask(__name__)



transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image, model, transform):
    # Load and preprocess the image
    image = image.convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        image = image.to("cpu")  # Move image to the same device as the model
        output = model(image)
        probabilities = F.softmax(output, dim=1)  # Apply softmax to get probabilities
        predicted_class = torch.argmax(probabilities, dim=1)
    return predicted_class.item()




@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # image_data = request.json['image']
    # # Convert image data to numpy array
    # model = torch.load('/efficentnetv2DoodleModel6.6.pth', map_location=torch.device("cpu"))

    # # Make prediction using your ML model
    # # prediction = predict_doodle(image_array)

    # # For demonstration, we'll return a dummy prediction

    # predicted_class = predict_image(image_data, model, transform)

    return jsonify({'prediction': "predicted_class"})


if __name__ == '__main__':
    app.run(debug=True)