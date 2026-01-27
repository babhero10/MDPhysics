"""
FastAPI web application for image deblurring.

Usage:
    export MODEL_CHECKPOINT=outputs/mdt_edited/.../best_model_epoch493.pt
    uvicorn src.web.app:app --host 0.0.0.0 --port 8000
"""

import base64
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
from starlette.requests import Request

# Import from predict.py
from src.predict import load_model, load_depth_model, compute_depth, postprocess_output

# Global state for loaded models
models = {
    "deblur": None,
    "depth": None,
    "device": None,
    "config": None,
}


def get_config():
    """Get configuration from environment variables."""
    return {
        "checkpoint": os.environ.get("MODEL_CHECKPOINT"),
        "config": os.environ.get("MODEL_CONFIG", "configs/model/mdt_edited.yaml"),
        "device": os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"),
        "enable_depth": os.environ.get("ENABLE_DEPTH", "false").lower() == "true",
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup."""
    config = get_config()

    if not config["checkpoint"]:
        print("WARNING: MODEL_CHECKPOINT not set. Model will not be loaded.")
        print("Set MODEL_CHECKPOINT environment variable to enable deblurring.")
        yield
        return

    checkpoint_path = Path(config["checkpoint"])
    if not checkpoint_path.exists():
        print(f"WARNING: Checkpoint not found: {checkpoint_path}")
        yield
        return

    config_path = Path(config["config"])
    if not config_path.exists():
        print(f"WARNING: Config not found: {config_path}")
        yield
        return

    # Setup device
    device = torch.device(config["device"])
    if config["device"] == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device("cpu")

    print(f"Using device: {device}")
    models["device"] = device
    models["config"] = config

    # Load deblurring model
    print(f"Loading deblur model from {checkpoint_path}")
    models["deblur"] = load_model(str(config_path), str(checkpoint_path), device)
    print("Deblur model loaded successfully")

    # Load depth model if enabled
    if config["enable_depth"]:
        print("Loading depth model...")
        models["depth"] = load_depth_model(device)
        print("Depth model loaded successfully")

    yield

    # Cleanup
    models["deblur"] = None
    models["depth"] = None
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


app = FastAPI(
    title="Image Deblurring API",
    description="Web interface for deblurring images using deep learning",
    lifespan=lifespan,
)

# Get the directory where this file is located
WEB_DIR = Path(__file__).parent

# Mount static files
app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")

# Setup templates
templates = Jinja2Templates(directory=WEB_DIR / "templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the main page."""
    config = get_config()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "model_loaded": models["deblur"] is not None,
            "depth_available": models["depth"] is not None,
            "depth_enabled": config.get("enable_depth", False),
        },
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    config = get_config()
    return {
        "status": "ok",
        "model_loaded": models["deblur"] is not None,
        "depth_available": models["depth"] is not None,
        "device": str(models["device"]) if models["device"] else None,
        "config": {
            "checkpoint": config["checkpoint"],
            "enable_depth": config["enable_depth"],
        },
    }


def preprocess_image_bytes(image_bytes: bytes, device: torch.device) -> torch.Tensor:
    """Load image from bytes and preprocess."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(img, dtype=np.uint8)
    tensor = torch.from_numpy(img_np).permute(2, 0, 1).float().div_(255.0)
    tensor = tensor.unsqueeze(0).to(device)
    return tensor


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


@app.post("/api/deblur")
async def deblur(
    file: UploadFile = File(...),
    use_depth: bool = Form(False),
):
    """
    Deblur an uploaded image.

    Args:
        file: The image file to deblur
        use_depth: Whether to use depth estimation (if available)

    Returns:
        JSON with base64-encoded original and deblurred images
    """
    if models["deblur"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Set MODEL_CHECKPOINT environment variable and restart server.",
        )

    # Validate file type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read image bytes
        image_bytes = await file.read()

        # Preprocess
        device = models["device"]
        input_tensor = preprocess_image_bytes(image_bytes, device)

        # Compute depth if requested and available
        depth = None
        if use_depth and models["depth"] is not None:
            depth = compute_depth(input_tensor, models["depth"], device)

        # Run inference
        device_type = "cuda" if device.type == "cuda" else "cpu"
        with torch.no_grad():
            with torch.amp.autocast(device_type):
                output = models["deblur"](input_tensor, depth=depth)
                pred = torch.clamp(output["sharp_image"], 0.0, 1.0)

        # Convert to images
        original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        deblurred_image = postprocess_output(pred)

        # Encode as base64
        original_b64 = image_to_base64(original_image)
        deblurred_b64 = image_to_base64(deblurred_image)

        return JSONResponse({
            "success": True,
            "original": f"data:image/png;base64,{original_b64}",
            "deblurred": f"data:image/png;base64,{deblurred_b64}",
            "used_depth": use_depth and models["depth"] is not None,
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
