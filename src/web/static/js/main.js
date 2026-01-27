// DOM Elements
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const depthToggle = document.getElementById('depthToggle');
const loading = document.getElementById('loading');
const results = document.getElementById('results');
const error = document.getElementById('error');
const errorMessage = document.getElementById('errorMessage');
const originalImage = document.getElementById('originalImage');
const deblurredImage = document.getElementById('deblurredImage');
const downloadBtn = document.getElementById('downloadBtn');
const newImageBtn = document.getElementById('newImageBtn');
const retryBtn = document.getElementById('retryBtn');

// State
let currentDeblurredData = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupDropzone();
    setupButtons();
});

function setupDropzone() {
    // Click to upload
    dropzone.addEventListener('click', () => fileInput.click());

    // File input change
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Drag events
    dropzone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropzone.classList.add('dragover');
    });

    dropzone.addEventListener('dragleave', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');
    });

    dropzone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropzone.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });
}

function setupButtons() {
    downloadBtn.addEventListener('click', downloadResult);
    newImageBtn.addEventListener('click', resetUI);
    retryBtn.addEventListener('click', resetUI);
}

async function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showError('Please select an image file');
        return;
    }

    // Show loading state
    showLoading();

    try {
        const result = await uploadAndProcess(file);
        showResults(result);
    } catch (err) {
        showError(err.message || 'An error occurred while processing the image');
    }
}

async function uploadAndProcess(file) {
    const formData = new FormData();
    formData.append('file', file);

    // Add depth option if toggle exists and is checked
    if (depthToggle) {
        formData.append('use_depth', depthToggle.checked);
    }

    const response = await fetch('/api/deblur', {
        method: 'POST',
        body: formData,
    });

    if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error: ${response.status}`);
    }

    return response.json();
}

function showLoading() {
    dropzone.parentElement.classList.add('hidden');
    results.classList.add('hidden');
    error.classList.add('hidden');
    loading.classList.remove('hidden');
}

function showResults(data) {
    loading.classList.add('hidden');

    // Set images
    originalImage.src = data.original;
    deblurredImage.src = data.deblurred;

    // Store for download
    currentDeblurredData = data.deblurred;

    results.classList.remove('hidden');
}

function showError(message) {
    loading.classList.add('hidden');
    results.classList.add('hidden');
    errorMessage.textContent = message;
    error.classList.remove('hidden');
}

function resetUI() {
    loading.classList.add('hidden');
    results.classList.add('hidden');
    error.classList.add('hidden');
    dropzone.parentElement.classList.remove('hidden');

    // Reset file input
    fileInput.value = '';
    currentDeblurredData = null;
}

function downloadResult() {
    if (!currentDeblurredData) return;

    // Create download link
    const link = document.createElement('a');
    link.href = currentDeblurredData;
    link.download = 'deblurred_image.png';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}
