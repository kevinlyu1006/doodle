const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clearBtn');
const guessBtn = document.getElementById('guessBtn');
const result = document.getElementById('result');

let isDrawing = false;

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

    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
}

// Event listeners
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

clearBtn.addEventListener('click', () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    result.textContent = '';
});

guessBtn.addEventListener('click', async () => {
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    const grayScaleData = [];

    for (let i = 0; i < imageData.length; i += 4) {
        const avg = (imageData[i] + imageData[i + 1] + imageData[i + 2]) / 3;
        grayScaleData.push(avg);
    }

    const response = await fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: grayScaleData }),
    });

    const data = await response.json();
    result.textContent = `I think it's a ${data.prediction}!`;
});