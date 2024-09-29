import os
from typing import Literal

import mmcv
from fastapi import FastAPI, File, UploadFile, HTTPException
from mmdet.apis import init_detector, inference_detector

app = FastAPI()

CONFIG_DIR = "mmdetection/configs/"
CHECKPOINT_DIR = "weights/"
DEVICE = "cuda:0"

MODEL_TYPES = {
    "faster_rcnn": "faster_rcnn/custom_faster_rcnn_r50_fpn.py",
    "cascade_rcnn": "cascade_rcnn/custom_cascade_rcnn_r50_fpn.py",
    # More model types
}

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
    model: Literal["faster_rcnn", "cascade_rcnn"] = "faster_rcnn",
    image: UploadFile = File(...)
):
    try:

        detector = load_model(model)

        img_content = await image.read()
        img = mmcv.imread(img_content)
        
        result = inference_detector(detector, img)
        
        processed_result = f"Processed {type} detection result"
        
        return {"result": processed_result}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3000)