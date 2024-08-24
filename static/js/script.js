const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clearBtn');
const predictionsList = document.getElementById('predictionsList');

let isDrawing = false;
let lastPredictionTime = 0;
const predictionInterval = 500; // Minimum time between predictions in milliseconds

// Set up canvas
ctx.strokeStyle = '#4a4543';
ctx.lineWidth = 2;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

// Drawing functions
function startDrawing(e) {
    isDrawing = true;
    draw(e);
}

function stopDrawing() {
    isDrawing = false;
    ctx.beginPath();
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

    // Check if enough time has passed since the last prediction
    if (Date.now() - lastPredictionTime > predictionInterval) {
        getPrediction();
    }
}

async function getPrediction() {
    lastPredictionTime = Date.now();
    const dataURL = canvas.toDataURL('image/png');

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: dataURL }),
        });

        const data = await response.json();
        updatePredictionsList(data.predictions);
    } catch (error) {
        console.error('Error getting prediction:', error);
    }
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
//static/js/script.js
// Event listeners for mouse
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

// Event listeners for touch
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