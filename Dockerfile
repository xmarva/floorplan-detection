FROM gcr.io/kaggle-gpu-images/python:v152

ADD . /kaggle/working

RUN pip install lmdb svgpathtools pycocotools openmim wandb fastapi
RUN mim install mmdet
RUN cd mmdetection && pip install -q -e .

# docker build -t florplan_detection .