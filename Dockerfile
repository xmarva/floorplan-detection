# Inference server image. PyTorch 2.1.2 with CUDA 12.1: mmcv 2.1.0 has prebuilt wheels for this pair.
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# OpenCV, which mmcv and mmengine install, needs these system libraries.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The configs inherit from mmdetection/configs/_base_. The mmdet package ships the same files, so link
# them instead of copying the submodule into the image.
RUN mkdir mmdetection \
    && ln -s "$(python -c 'import mmdet, os; print(os.path.dirname(mmdet.__file__))')/.mim/configs" mmdetection/configs
COPY run.py .
COPY configs configs

# Weights are not part of the image: mount the weights/ directory at /app/weights.
EXPOSE 5000
ENTRYPOINT ["python", "run.py"]
CMD ["--model", "cascade_swin"]
