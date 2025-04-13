# src/train_detr.py

import torch
from torch.utils.data import DataLoader
from src.dataset import YoloFishDataset
from src.models.detr_builder import build_detr_model
from transformers import DetrImageProcessor
import os
from tqdm import tqdm

def yolo_to_coco_format(yolo_boxes, image_size):
    """
    Converts YOLO format boxes to COCO-style format:
    - YOLO: [x_center, y_center, width, height] (all relative)
    - COCO: [x_min, y_min, width, height] (absolute pixels)
    """
    boxes = []
    for box in yolo_boxes:
        _, xc, yc, w, h = box.tolist()  # Ignore class id
        x_min = (xc - w / 2) * image_size
        y_min = (yc - h / 2) * image_size
        width = w * image_size
        height = h * image_size
        boxes.append([x_min, y_min, width, height])
    return boxes

def train_detr(train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir, epochs=20, lr=1e-4, img_size=640):
    """
    Fine-tunes DETR model on fish sonar dataset using HuggingFace Transformers.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize dataset and loader
    train_dataset = YoloFishDataset(train_img_dir, train_lbl_dir, img_size=img_size)
    val_dataset = YoloFishDataset(val_img_dir, val_lbl_dir, img_size=img_size)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)

    # Load model and processor
    model, processor, optimizer = build_detr_model(num_classes=2, lr=lr)
    model.to(device)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        print(f"\n[Epoch {epoch+1}/{epochs}]")

        for batch in tqdm(train_loader, desc="Training"):
            pixel_values = []
            targets = []

            for i in range(len(batch['image'])):
                image = batch['image'][i].squeeze(0).numpy()  # shape: [H, W]
                image_3ch = torch.tensor([image, image, image])  # to 3-channel
                pixel_values.append(image_3ch)

                boxes = batch['labels'][i]
                if len(boxes) == 0:
                    target = {"class_labels": torch.empty(0, dtype=torch.long),
                              "boxes": torch.empty((0, 4), dtype=torch.float)}
                else:
                    boxes_coco = yolo_to_coco_format(boxes, image_size=img_size)
                    target = {
                        "class_labels": torch.zeros(len(boxes), dtype=torch.long),  # all 'fish' = class 0
                        "boxes": torch.tensor(boxes_coco, dtype=torch.float)
                    }
                targets.append(target)

            # Process inputs using DetrImageProcessor
            inputs = processor(images=pixel_values, return_tensors="pt")
            inputs["pixel_values"] = inputs["pixel_values"].to(device)

            # Prepare labels in processor-compatible format
            target_inputs = []
            for t in targets:
                target_inputs.append({
                    "class_labels": t["class_labels"].to(device),
                    "boxes": t["boxes"].to(device)
                })

            outputs = model(**inputs, labels=target_inputs)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "weights/detr_finetuned.pth")
    print("[INFO] DETR model saved to weights/detr_finetuned.pth")
