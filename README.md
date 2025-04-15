# STA2453 Project: Automating Sonar Image Analysis in Ontario Lakes for Fish Counting

This project implements a complete object detection pipeline to identify and localize fish in underwater sonar video footage using both **YOLOv8** and **DETR (DEtection TRansformer)**.

---

## Overview

I fine-tune two state-of-the-art models:

- **YOLOv8** (from Ultralytics) for fast, efficient fish detection  
- **DETR** (from HuggingFace Transformers) for transformer-based detection on sonar images

### The pipeline includes:

- Preprocessing and frame extraction from sonar video  
- Parsing YOLO-style bounding box labels  
- Dataset splitting and loading  
- Training both YOLO and DETR on the Alaska dataset  
- Transfer learning/fine-tuning on Ontario footage  
- Evaluation and visualization  

---

## Directory Structure

```bash
STA2543Project/
├── src/                  # All source code
│   ├── dataset.py        # YoloFishDataset class
│   ├── preprocessing.py  # Video-to-frames, label processing, dataset split
│   ├── train_yolo.py     # YOLOv8 training script
│   ├── train_detr.py     # DETR training script
│   └── models/           # DETR model builder (optional)
├── weights/              # Trained models will be saved here
├── data/                 # Raw and preprocessed Alaska/Ontario datasets
├── requirements.txt      # All dependencies
└── README.md             # This file

```

##  Quickstart (Google Colab Demo)

To reproduce these results, use the following Colab notebook:  
 **[Run the notebook here](https://colab.research.google.com/drive/1_9YTcMLis6IeCAAgTZuVTFCOTF7_omZi?usp=sharing)**

It walks through:

- Downloading and preparing data  
- Extracting video frames   
- Pretraining and fine-tuning DETR and YOLOv8 

---

## Datasets

### Alaska Dataset
- Pre-extracted frames  
- YOLO-format labels:  
  `[class, x_center, y_center, width, height]`

### Ontario Dataset
- Raw `.mp4` video  
- Labels as `.txt` files with matching format  

> Both are organized and split using the `preprocessing.py` pipeline.

---

## Setup

To run locally or on Colab:

```bash
pip install -r requirements.txt
```
##  Dependencies include:

- `ultralytics`
- `transformers`
- `torch`
- `opencv-python`
- `tqdm`

## **Contributors**
Anandita Mahika
