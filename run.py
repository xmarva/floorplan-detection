import os
import json
import argparse
import numpy as np
from typing import Dict, Any
import mmcv
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from mmdet.apis import init_detector, inference_detector

app = FastAPI()

CONFIG_DIR = "configs/"
CHECKPOINT_DIR = "weights/"
DEVICE = "cuda:0"

MODEL_TYPES = {
    "cascade_swin": "cascade_swin.py",
    "faster_rcnn": "faster_rcnn.py",
    "retinanet": "retinanet.py",
    # More models
}

global_model = None

def load_model(model_type: str):
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unsupported model type: {model_type}")
    config_file = os.path.join(CONFIG_DIR, MODEL_TYPES[model_type])
    checkpoint_file = os.path.join(CHECKPOINT_DIR, f"{model_type}_latest.pth")
    if not os.path.exists(config_file) or not os.path.exists(checkpoint_file):
        raise FileNotFoundError(f"Config or checkpoint file not found for {model_type}")
    return init_detector(config_file, checkpoint_file, device=DEVICE)

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
    try:
        global global_model
        if global_model is None:
            raise ValueError("Model not initialized. Please restart the server with the correct model type.")
        img_content = await image.read()
        img = mmcv.imread(img_content)
        result = inference_detector(global_model, img)

        processed_result = process_inference_result(result)
        return processed_result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def main():
    parser = argparse.ArgumentParser(description="Run MMDetection inference API")
    
    parser.add_argument("--model", type=str, choices=MODEL_TYPES.keys(), required=True,
                        help="Type of model to use for inference")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the API on")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the API on")

    args = parser.parse_args()

    global global_model
    global_model = load_model(args.model)
    print(f"Starting server with {args.model} model...")
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()