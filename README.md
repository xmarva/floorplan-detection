# ❌ FloorPlan Detection (Not Finished)

![FloorPlan](examples/image.png)

## Overview

This repository contains the API code for detecting walls and rooms in architectural floor plans. 
I’m also sharing my notebooks, research results, and some thoughts on how the results could be improved.

## How to run

### Model Weights
Download the  [pre-trained weights](https://drive.google.com/file/d/1A9sZlM8NXe2bA7jgzm7jNeiB6oXjDUP-/view?usp=sharing) and place them in the `/models` directory. You can adjust model types by specifying them when running the server.

### Build with Docker
To build the code environment, use the provided Dockerfile in the root directory:
`docker build -t floorplan-detection .`

### Start the Server
To start the FastAPI server, use the following command:
`python run.py --model <model_name>`

## Info

### First things first
The task doesn’t come with labeled data or evaluation criteria, so I treated it as an open-ended project. The provided floor plan examples are inconsistent—using different labels, line thicknesses, and so on—so I figured neural network-based methods would be the best approach here.

### Dataset
I’m using the `CubiCasa5k` dataset. This dataset provides annotations for different room types, but I’m focusing on two categories: Walls and Rooms (without further subclassification). The dataset initially comes with masks, but I use only bboxes. I’ve converted it to `COCO` format for smoother integration with detection frameworks. 

If you want to use [original dataset](https://zenodo.org/records/2613548), you need to download it and put into `data/CubiCasa5k/data` directory. For a quick overview of the dataset, check out the notebook: `/notebooks/cubicasa5k_dataset.ipynb`
If you want to use my [CubiCasa5k_COCO](https://drive.google.com/drive/folders/1hKRWrP-ZKk6ZHrjHOSRSxPe_r_kMd8uh?usp=sharing) dataset of train another model on it, you need to download it and puth into `data/cubicasa5k_coco` directory. You still need images in `data/CubiCasa5k/data`!
If you want to modify original data to coco format (with masks or anything else) you can use `data/cubicasa5k_to_coco.py` script.