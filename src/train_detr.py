# src/train_detr.py

import torch
from torch.utils.data import DataLoader
from src.dataset import YoloFishDataset
from src.models.detr_builder import build_detr_model
from tqdm import tqdm

def yolo_to_coco_format(yolo_boxes, image_size):
    boxes = []
    for box in yolo_boxes:
        _, xc, yc, w, h = box.tolist()
        x_min = (xc - w / 2) * image_size
        y_min = (yc - h / 2) * image_size
        width = w * image_size
        height = h * image_size
        boxes.append([x_min, y_min, width, height])
    return boxes

def train_detr(train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir,
               epochs=20, lr=1e-4, img_size=640, weight_path=None, save_path="weights/detr_finetuned.pth"):
    """
    Fine-tunes or trains DETR model using HuggingFace Transformers.
    Supports loading from a pretrained checkpoint.

    Args:
        weight_path (str): Optional path to pretrained .pth weights.
        save_path (str): Where to save the fine-tuned weights.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_dataset = YoloFishDataset(train_img_dir, train_lbl_dir, img_size=img_size)
    val_dataset = YoloFishDataset(val_img_dir, val_lbl_dir, img_size=img_size)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # Build model
    model, processor, optimizer = build_detr_model(num_classes=2, lr=lr)
    if weight_path:
        model.load_state_dict(torch.load(weight_path))
        print(f"[INFO] Loaded pretrained weights from {weight_path}")
    model.to(device)

    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        print(f"\n[Epoch {epoch+1}/{epochs}]")

        for batch in tqdm(train_loader, desc="Training"):
            pixel_values = []
            targets = []

            for i in range(len(batch['image'])):
                image = batch['image'][i].squeeze(0).numpy()
                image_3ch = torch.tensor([image, image, image])
                pixel_values.append(image_3ch)

                boxes = batch['labels'][i]
                if len(boxes) == 0:
                    targets.append({
                        "class_labels": torch.empty(0, dtype=torch.long),
                        "boxes": torch.empty((0, 4), dtype=torch.float)
                    })
                else:
                    coco_boxes = yolo_to_coco_format(boxes, img_size)
                    targets.append({
                        "class_labels": torch.zeros(len(boxes), dtype=torch.long),
                        "boxes": torch.tensor(coco_boxes, dtype=torch.float)
                    })

            inputs = processor(images=pixel_values, return_tensors="pt")
            inputs["pixel_values"] = inputs["pixel_values"].to(device)
            target_inputs = [{"class_labels": t["class_labels"].to(device),
                              "boxes": t["boxes"].to(device)} for t in targets]

            outputs = model(**inputs, labels=target_inputs)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"[INFO] DETR model saved to {save_path}")
