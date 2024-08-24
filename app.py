import io
from flask import Flask, render_template, request, jsonify
# from your_model_file import predict_doodle
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import base64


app = Flask(__name__)


indexToLabel = {0: 'traffic light',
 1: 'bed',
 2: 'van',
 3: 'laptop',
 4: 'tractor',
 5: 'windmill',
 6: 'square',
 7: 'pineapple',
 8: 'candle',
 9: 'mosquito',
 10: 'pear',
 11: 'boomerang',
 12: 'lollipop',
 13: 'waterslide',
 14: 'swan',
 15: 'triangle',
 16: 'diving board',
 17: 'crayon',
 18: 'hockey puck',
 19: 'moustache',
 20: 'calendar',
 21: 'cow',
 22: 'fire hydrant',
 23: 'hot air balloon',
 24: 'helmet',
 25: 'parrot',
 26: 'hot tub',
 27: 'baseball',
 28: 'saw',
 29: 'mouth',
 30: 'passport',
 31: 'campfire',
 32: 'car',
 33: 'bulldozer',
 34: 'pencil',
 35: 'wine glass',
 36: 'marker',
 37: 'axe',
 38: 'mug',
 39: 'foot',
 40: 'door',
 41: 'beach',
 42: 'cruise ship',
 43: 'drums',
 44: 'necklace',
 45: 'spoon',
 46: 'motorbike',
 47: 'megaphone',
 48: 'penguin',
 49: 'washing machine',
 50: 'giraffe',
 51: 'monkey',
 52: 'shoe',
 53: 'microphone',
 54: 'skyscraper',
 55: 'blackberry',
 56: 'sword',
 57: 'nail',
 58: 'birthday cake',
 59: 'carrot',
 60: 'lobster',
 61: 'hourglass',
 62: 'microwave',
 63: 'cannon',
 64: 'clarinet',
 65: 'basketball',
 66: 'pliers',
 67: 'bee',
 68: 'flashlight',
 69: 'leaf',
 70: 'belt',
 71: 'grass',
 72: 'river',
 73: 'peas',
 74: 'elbow',
 75: 'tiger',
 76: 'roller coaster',
 77: 'piano',
 78: 'trumpet',
 79: 'snowflake',
 80: 'bandage',
 81: 'bowtie',
 82: 'harp',
 83: 'onion',
 84: 'stairs',
 85: 'bus',
 86: 'oven',
 87: 'stop sign',
 88: 'chair',
 89: 'guitar',
 90: 'headphones',
 91: 'hockey stick',
 92: 'sheep',
 93: 'leg',
 94: 'popsicle',
 95: 'suitcase',
 96: 'snorkel',
 97: 'angel',
 98: 'scissors',
 99: 'rabbit',
 100: 'butterfly',
 101: 'bear',
 102: 'dog',
 103: 'whale',
 104: 'frog',
 105: 'cat',
 106: 'elephant',
 107: 'bird',
 108: 'fish'}

model = torch.jit.load('efficentnetv2DoodleModel6.6_scripted.pt')
model.eval()


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
    predictions = []
    with torch.no_grad():
        image = image.to("cpu")  # Move image to the same device as the model
        output = model(image)
        probabilities = F.softmax(output, dim=1)  # Apply softmax to get probabilities
        top5_values, top5_indices = torch.topk(probabilities, 5, dim=1)
        predictions = []
        for i in range(5):
            predictions.append({
                "label": indexToLabel[top5_indices[0, i].item()],  # Assuming batch size is 1
                "probability": top5_values[0, i].item()            # Get the probability
            })
    return predictions




@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_data = data['image']
    header, encoded = image_data.split(',', 1)
    image_bytes = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # Convert image to grayscale
    gray_image = image.convert('L')
    threshold = 1  # Adjust this value if needed
    binary_image = gray_image.point(lambda p: 255 if p > threshold else 0)

    # Create a new image with a white background
    background_color = (255, 255, 255)  # White background
    new_image = Image.new('RGB', image.size, background_color)
    black_mask = binary_image.convert('L')
    new_image.paste((0, 0, 0), (0, 0, image.size[0], image.size[1]), black_mask)

    # Make prediction
    predictions = predict_image(new_image, model, transform)

    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run(debug=True)