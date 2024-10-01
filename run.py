import os
import json
import argparse
import numpy as np
from typing import Dict, Any
import mmcv
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from mmdet.apis import init_detector, inference_detector
import cv2
import torch
import logging

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_DIR = "configs/"
CHECKPOINT_DIR = "weights/"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

MODEL_TYPES = {
    "cascade_swin": "cascade_swin.py",
    "faster_rcnn": "faster_rcnn.py",
    "retinanet": "retinanet.py",
    # More models here
}

global_model = None

def load_model(model_type: str):
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unsupported model type: {model_type}")
    config_file = os.path.join(CONFIG_DIR, MODEL_TYPES[model_type])
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"{model_type}_latest.pth")
    if not os.path.exists(config_file) or not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Config or checkpoint file not found for {model_type}")
    try:
        model = init_detector(config_file, checkpoint_file, device=DEVICE)
        logger.info(f"Model {model_type} loaded successfully on {DEVICE}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model {model_type}: {str(e)}")
        raise

def process_inference_result(result) -> Dict[str, Any]:
    bboxes = result.pred_instances.bboxes.cpu().numpy()
    labels = result.pred_instances.labels.cpu().numpy()
    scores = result.pred_instances.scores.cpu().numpy()
    walls = []
    rooms = []
    for i, (bbox, label, score) in enumerate(zip(bboxes, labels, scores)):
        x1, y1, x2, y2 = bbox
        item = {
            "id": f"{'wall' if label == 0 else 'room'}_{i+1}",
            "position": {
                "start": {"x": float(x1), "y": float(y1)},
                "end": {"x": float(x2), "y": float(y2)}
            },
            "confidence": float(score)
        }
        if label == 0:
            walls.append(item)
        else:
            rooms.append(item)
    return {
        "type": "floor_plan",
        "confidence": float(np.mean(scores)),  # Overall confidence as mean of all scores
        "detectionResults": {
            "walls": walls,
            "rooms": rooms
        }
    }

@app.post("/run-inference")
async def run_inference(image: UploadFile = File(...)):
    if not global_model:
        raise HTTPException(status_code=500, detail="Model not initialized. Please restart the server with the correct model type.")
    
    if image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Only JPEG and PNG images are supported.")
    
    try:
        contents = await image.read()
        if len(contents) > MAX_FILE_SIZE:
            raise HTTPException(status_code=400, detail="File size exceeds the maximum limit of 10 MB.")
        
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Failed to decode image. Please ensure the image is not corrupted.")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        result = inference_detector(global_model, img)
        processed_result = process_inference_result(result)
        logger.info("Inference completed successfully")
        return processed_result
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred during inference: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Run MMDetection inference API")
    parser.add_argument("--model", type=str, choices=MODEL_TYPES.keys(), required=True,
                        help="Type of model to use for inference")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the API on")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the API on")
    args = parser.parse_args()

    global global_model
    try:
        global_model = load_model(args.model)
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        return

    logger.info(f"Starting server with {args.model} model on device: {DEVICE}")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()