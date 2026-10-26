"""HTTP API that finds walls and rooms in a floor plan image with an MMDetection model.

Usage:
    python run.py --model cascade_swin [--checkpoint PATH] [--device cpu] [--score-threshold 0.3]

The server loads the model once at startup and serves POST /run-inference (multipart field `image`,
a JPEG or PNG file) and GET /health. See README.md for the response format.
"""

import argparse
import logging
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "configs"
WEIGHTS_DIR = ROOT / "weights"

# Model name -> config file in configs/. Weights are expected at weights/<name>_latest.pth.
MODELS = {
    "cascade_swin": "cascade_swin.py",  # Cascade R-CNN, ResNet-101 + FPN (the name is historical)
    "faster_rcnn": "faster_rcnn.py",
    "retinanet": "retinanet.py",
}
CLASS_NAMES = ("wall", "room")  # label order used by all configs and checkpoints
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # bytes

logger = logging.getLogger("floorplan")


class Point(BaseModel):
    x: float
    y: float


class Position(BaseModel):
    start: Point  # top-left corner of the bounding box, in pixels of the input image
    end: Point  # bottom-right corner


class Detection(BaseModel):
    id: str
    position: Position
    confidence: float


class DetectionResults(BaseModel):
    walls: list[Detection]
    rooms: list[Detection]


class InferenceResponse(BaseModel):
    type: str = "floor_plan"
    confidence: float  # mean confidence of the returned detections, 0 when there are none
    detectionResults: DetectionResults  # camelCase name kept: it is part of the API


def build_response(bboxes: np.ndarray, labels: np.ndarray, scores: np.ndarray) -> InferenceResponse:
    """Group (x1, y1, x2, y2) boxes by class and number them per class: wall_1, wall_2, ..., room_1, ..."""
    groups: dict[str, list[Detection]] = {name: [] for name in CLASS_NAMES}
    for (x1, y1, x2, y2), label, score in zip(bboxes, labels, scores, strict=True):
        name = CLASS_NAMES[int(label)]
        groups[name].append(
            Detection(
                id=f"{name}_{len(groups[name]) + 1}",
                position=Position(start=Point(x=float(x1), y=float(y1)), end=Point(x=float(x2), y=float(y2))),
                confidence=float(score),
            )
        )
    return InferenceResponse(
        confidence=float(np.mean(scores)) if len(scores) else 0.0,
        detectionResults=DetectionResults(walls=groups["wall"], rooms=groups["room"]),
    )


class Detector:
    """An MMDetection model plus the score threshold applied to its output."""

    def __init__(self, config: Path, checkpoint: Path, device: str, score_threshold: float):
        # Imported here so that `python run.py --help` does not wait for torch and mmdet.
        from mmdet.apis import inference_detector, init_detector

        self._inference_detector = inference_detector
        self.model = init_detector(str(config), str(checkpoint), device=device)
        self.device = device
        self.score_threshold = score_threshold
        self._lock = threading.Lock()  # one inference at a time on the shared model

    def predict(self, image_bgr: np.ndarray) -> InferenceResponse:
        # MMDetection expects BGR arrays, as returned by cv2; the model converts them to RGB itself.
        with self._lock:
            result = self._inference_detector(self.model, image_bgr)
        instances = result.pred_instances
        keep = instances.scores >= self.score_threshold
        return build_response(
            instances.bboxes[keep].cpu().numpy(),
            instances.labels[keep].cpu().numpy(),
            instances.scores[keep].cpu().numpy(),
        )


def create_app(detector: Detector, model_name: str) -> FastAPI:
    app = FastAPI(title="Floor plan detection", description="Walls and rooms in floor plan images.")

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model": model_name, "device": detector.device}

    # A plain `def` endpoint runs in FastAPI's thread pool, so inference does not block the event loop.
    @app.post("/run-inference", response_model=InferenceResponse)
    def run_inference(image: UploadFile = File(...)) -> InferenceResponse:
        if image.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(status_code=415, detail="Only JPEG and PNG images are supported.")
        data = image.file.read(MAX_FILE_SIZE + 1)
        if len(data) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail=f"The image is larger than {MAX_FILE_SIZE // 2**20} MB.")
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode the image.")
        try:
            response = detector.predict(img)
        except Exception:
            logger.exception("Inference failed for %s", image.filename)
            raise HTTPException(status_code=500, detail="Inference failed, see the server log.") from None
        logger.info(
            "%s: %d walls, %d rooms",
            image.filename,
            len(response.detectionResults.walls),
            len(response.detectionResults.rooms),
        )
        return response

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a model that finds walls and rooms in floor plans.")
    parser.add_argument("--model", choices=MODELS, required=True, help="Model to serve.")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint file (default: weights/<model>_latest.pth).")
    parser.add_argument("--device", help="Torch device, e.g. cuda:0 or cpu (default: cuda:0 if available, else cpu).")
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.3,
        help="Drop detections with a lower confidence (default: 0.3; 0 keeps everything the model returns).",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Address to bind (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on (default: 5000).")
    args = parser.parse_args()
    if not 0.0 <= args.score_threshold <= 1.0:
        parser.error("--score-threshold must be between 0 and 1")
    return args


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    config = CONFIG_DIR / MODELS[args.model]
    checkpoint = args.checkpoint or WEIGHTS_DIR / f"{args.model}_latest.pth"
    if not checkpoint.is_file():
        sys.exit(f"Checkpoint not found: {checkpoint}\nDownload the weights (see README.md) into {WEIGHTS_DIR}/.")

    if args.device is None:
        import torch

        args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    logger.info("Loading %s from %s on %s", args.model, checkpoint, args.device)
    detector = Detector(config, checkpoint, args.device, args.score_threshold)
    uvicorn.run(create_app(detector, args.model), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
