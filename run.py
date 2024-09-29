import os
import argparse
from typing import Literal

import mmcv
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from mmdet.apis import init_detector, inference_detector

app = FastAPI()

CONFIG_DIR = "configs/"
CHECKPOINT_DIR = "weights/"
DEVICE = "cuda:0"

MODEL_TYPES = {
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
    
    return init_detector(config_file, checkpoint_file, device=DEVICE)

@app.post("/run-inference")
async def run_inference(
    type: Literal["wall", "room"],
    image: UploadFile = File(...)
):
    try:

        global global_model
        
        if global_model is None:
            raise ValueError("Model not initialized. Please restart the server with the correct model type.")
        
        img_content = await image.read()
        img = mmcv.imread(img_content)
        
        result = inference_detector(global_model, img)
        
        processed_result = f"Processed {type} detection result"
        
        return {"result": processed_result}
    
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