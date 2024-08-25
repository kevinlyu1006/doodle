const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clearBtn');
const predictionsList = document.getElementById('predictionsList');

let isDrawing = false;
let session;
let lastPredictionTime = 0;
let predictionInterval = 500;
const indexToLabel = {
    0: 'traffic light', 1: 'bed', 2: 'van', 3: 'laptop', 4: 'tractor',
    5: 'windmill', 6: 'square', 7: 'pineapple', 8: 'candle', 9: 'mosquito',
    10: 'pear', 11: 'boomerang', 12: 'lollipop', 13: 'waterslide', 14: 'swan',
    15: 'triangle', 16: 'diving board', 17: 'crayon', 18: 'hockey puck', 19: 'moustache',
    20: 'calendar', 21: 'cow', 22: 'fire hydrant', 23: 'hot air balloon', 24: 'helmet',
    25: 'parrot', 26: 'hot tub', 27: 'baseball', 28: 'saw', 29: 'mouth',
    30: 'passport', 31: 'campfire', 32: 'car', 33: 'bulldozer', 34: 'pencil',
    35: 'wine glass', 36: 'marker', 37: 'axe', 38: 'mug', 39: 'foot',
    40: 'door', 41: 'beach', 42: 'cruise ship', 43: 'drums', 44: 'necklace',
    45: 'spoon', 46: 'motorbike', 47: 'megaphone', 48: 'penguin', 49: 'washing machine',
    50: 'giraffe', 51: 'monkey', 52: 'shoe', 53: 'microphone', 54: 'skyscraper',
    55: 'blackberry', 56: 'sword', 57: 'nail', 58: 'birthday cake', 59: 'carrot',
    60: 'lobster', 61: 'hourglass', 62: 'microwave', 63: 'cannon', 64: 'clarinet',
    65: 'basketball', 66: 'pliers', 67: 'bee', 68: 'flashlight', 69: 'leaf',
    70: 'belt', 71: 'grass', 72: 'river', 73: 'peas', 74: 'elbow',
    75: 'tiger', 76: 'roller coaster', 77: 'piano', 78: 'trumpet', 79: 'snowflake',
    80: 'bandage', 81: 'bowtie', 82: 'harp', 83: 'onion', 84: 'stairs',
    85: 'bus', 86: 'oven', 87: 'stop sign', 88: 'chair', 89: 'guitar',
    90: 'headphones', 91: 'hockey stick', 92: 'sheep', 93: 'leg', 94: 'popsicle',
    95: 'suitcase', 96: 'snorkel', 97: 'angel', 98: 'scissors', 99: 'rabbit',
    100: 'butterfly', 101: 'bear', 102: 'dog', 103: 'whale', 104: 'frog',
    105: 'cat', 106: 'elephant', 107: 'bird', 108: 'fish'
};

// Set up canvas
ctx.strokeStyle = '#4a4543';
ctx.lineWidth = 2;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

// Load ONNX model
async function loadModel() {
    try {
        const response = await fetch('/static/modelidk.onnx');
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const arrayBuffer = await response.arrayBuffer();
        const rawModel = new Uint8Array(arrayBuffer);
        
        // Create ONNX inference session
        session = await ort.InferenceSession.create(rawModel);
        console.log('Model loaded successfully');
    } catch (error) {
        console.error('Error loading the model:', error);
    }
}

// Drawing functions
function startDrawing(e) {
    isDrawing = true;
    draw(e);
}

function stopDrawing() {
    isDrawing = false;
    ctx.beginPath();
    getPrediction();
}

function draw(e) {
    if (!isDrawing) return;

    let x, y;
    const rect = canvas.getBoundingClientRect();

    if (e.type.startsWith('mouse')) {
        x = e.clientX - rect.left;
        y = e.clientY - rect.top;
    } else if (e.type.startsWith('touch')) {
        x = e.touches[0].clientX - rect.left;
        y = e.touches[0].clientY - rect.top;
    }

    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
    if (Date.now() - lastPredictionTime > predictionInterval) {
        getPrediction();
    }
}

async function getPrediction() {
    lastPredictionTime = Date.now();



    const dataURL = canvas.toDataURL('image/png');

    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: dataURL }),
    });

    const data = await response.json();
    // Fill the background with white
   
    const tensorList = data.tensor;
    
        // Reshape the array to match the original tensor shape
    const tensorShape = [1, 3, 256, 256];
    console.log(tensorList)

    const tensor = new ort.Tensor('float32', tensorList, tensorShape);
    console.log("d")

    // Run inference
    const outputMap = await session.run({ 'input': tensor });
    const output = outputMap['output'];

    // Post-process the output
    const predictions = postprocess(output);
    updatePredictionsList(predictions);
}



const mean = [0.485, 0.456, 0.406]; // Example mean values for RGB
const std = [0.229, 0.224, 0.225];  // Example std values for RGB

function preprocess(imageData) {
    // Convert the image data to a Float32Array
    // Assumes imageData is an ImageData object with RGBA data
    // Normalize pixel values to the range [0, 1]
    const normalizedData = new Float32Array(imageData.data.length / 4 * 3);

    for (let i = 0, j = 0; i < imageData.data.length; i += 4, j += 3) {
        normalizedData[j] = (imageData.data[i] / 255 - mean[0]) / std[0];        // Red channel
        normalizedData[j + 1] = (imageData.data[i + 1] / 255 - mean[1]) / std[1]; // Green channel
        normalizedData[j + 2] = (imageData.data[i + 2] / 255 - mean[2]) / std[2]; // Blue channel
    }

    // Convert the normalized data to a tensor
    const tensor = new ort.Tensor('float32', normalizedData, [1, 3, 256, 256]);

    return tensor;
}

function postprocess(output) {
    // Convert the output to probabilities and get top 5 predictions
    const probabilities = softmax(output.data);
    const topK = 5;
    const indices = Array.from(probabilities.keys())
        .sort((a, b) => probabilities[b] - probabilities[a])
        .slice(0, topK);

    return indices.map(index => ({
        label: indexToLabel[index],
        probability: probabilities[index]
    }));
}

function softmax(arr) {
    const max = Math.max(...arr);
    const expValues = arr.map(val => Math.exp(val - max));
    const sumExp = expValues.reduce((acc, val) => acc + val, 0);
    return expValues.map(val => val / sumExp);
}

function updatePredictionsList(predictions) {
    predictionsList.innerHTML = '';
    predictions.forEach(prediction => {
        const li = document.createElement('li');
        li.className = 'bg-slambook-button p-3 rounded-lg flex justify-between items-center';
        li.innerHTML = `
            <span class="font-semibold">${prediction.label}</span>
            <span class="bg-slambook-accent text-white px-2 py-1 rounded-full text-xs">
                ${(prediction.probability * 100).toFixed(2)}%
            </span>
        `;
        predictionsList.appendChild(li);
    });
}

// Event listeners
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

canvas.addEventListener('touchstart', startDrawing);
canvas.addEventListener('touchmove', draw);
canvas.addEventListener('touchend', stopDrawing);

clearBtn.addEventListener('click', () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    predictionsList.innerHTML = '<li class="bg-slambook-button p-3 rounded-lg"><span class="font-semibold">Drawing something...</span></li>';
});

// Prevent scrolling when touching the canvas
document.body.addEventListener("touchstart", function (e) {
    if (e.target == canvas) {
        e.preventDefault();
    }
}, { passive: false });
document.body.addEventListener("touchend", function (e) {
    if (e.target == canvas) {
        e.preventDefault();
    }
}, { passive: false });
document.body.addEventListener("touchmove", function (e) {
    if (e.target == canvas) {
        e.preventDefault();
    }
}, { passive: false });

// Load the model when the page loads
window.addEventListener('load', loadModel);
