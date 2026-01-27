"""
FastAPI web application for image deblurring.

Usage:
    uvicorn src.web.app:app --host 0.0.0.0 --port 8000
"""

import sys
from pathlib import Path

# Add src to path so Hydra can find model.MDT_Edited
SRC_DIR = Path(__file__).parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import base64
import io
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from omegaconf import OmegaConf
from PIL import Image
from starlette.requests import Request

from src.predict import load_config, load_models, compute_depth, postprocess_output

# Global state for loaded models
state = {
    "model": None,
    "depth_model": None,
    "device": None,
    "cfg": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models at startup."""
    try:
        cfg = load_config()
    except Exception as e:
        print(f"WARNING: Failed to load config: {e}")
        yield
        return

    # Setup device from config
    device = torch.device(cfg.device)
    if "cuda" in cfg.device and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device("cpu")

    print(f"Using device: {device}")
    state["device"] = device
    state["cfg"] = cfg

    # Load models
    print("Loading models...")
    try:
        model, depth_model = load_models(cfg, device)
        state["model"] = model
        state["depth_model"] = depth_model
        print("Models loaded successfully")
    except Exception as e:
        print(f"WARNING: Failed to load models: {e}")
        yield
        return

    yield

    # Cleanup
    state["model"] = None
    state["depth_model"] = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
    use_depth = False
    if state["cfg"]:
        use_depth = OmegaConf.select(state["cfg"], "dataset.use_depth", default=False)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "model_loaded": state["model"] is not None,
            "depth_available": state["depth_model"] is not None,
            "depth_enabled": use_depth,
        },
    )


@app.get("/health")
async def health():
    """Health check endpoint."""
    use_depth = False
    checkpoint = None
    if state["cfg"]:
        use_depth = OmegaConf.select(state["cfg"], "dataset.use_depth", default=False)
        checkpoint = OmegaConf.select(state["cfg"], "train.last_checkpoint", default=None)

    return {
        "status": "ok",
        "model_loaded": state["model"] is not None,
        "depth_available": state["depth_model"] is not None,
        "device": str(state["device"]) if state["device"] else None,
        "config": {
            "checkpoint": checkpoint,
            "use_depth": use_depth,
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
    if state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check config and restart server.",
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        image_bytes = await file.read()

        device = state["device"]
        input_tensor = preprocess_image_bytes(image_bytes, device)

        depth = None
        if use_depth and state["depth_model"] is not None:
            depth = compute_depth(input_tensor, state["depth_model"], device)

        device_type = "cuda" if device.type == "cuda" else "cpu"
        with torch.no_grad():
            with torch.amp.autocast(device_type):
                output = state["model"](input_tensor, depth=depth)
                pred = torch.clamp(output["sharp_image"], 0.0, 1.0)

        original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        deblurred_image = postprocess_output(pred)

        original_b64 = image_to_base64(original_image)
        deblurred_b64 = image_to_base64(deblurred_image)

        return JSONResponse({
            "success": True,
            "original": f"data:image/png;base64,{original_b64}",
            "deblurred": f"data:image/png;base64,{deblurred_b64}",
            "used_depth": use_depth and state["depth_model"] is not None,
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
