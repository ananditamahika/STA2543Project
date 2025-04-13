# src/train_yolo.py

from ultralytics import YOLO
import os

def train_yolo(yaml_path, model_size="yolov8n.pt", epochs=30, imgsz=640):
    """
    Trains YOLOv8 using the Ultralytics CLI API.
    Args:
        yaml_path (str): Path to dataset YAML file.
        model_size (str): Which YOLOv8 variant to use.
        epochs (int): Number of training epochs.
        imgsz (int): Image size.
    """
    print(f"[INFO] Starting YOLOv8 training using {model_size} on {yaml_path}")
    model = YOLO(model_size)

    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=imgsz,
        batch=16,
        name="yolo_fish_detection"
    )
    print(f"[INFO] Training complete. Results saved to: {results.save_dir}")
