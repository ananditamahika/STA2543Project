# src/dataset.py

import os
import torch
from torch.utils.data import Dataset
import cv2

class YoloFishDataset(Dataset):
    """
    PyTorch dataset for YOLO/DETR fish detection on grayscale sonar images.
    Expects YOLO-format annotations.
    """
    def __init__(self, image_dir, label_dir, transform=None, img_size=640):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.img_size = img_size
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name.replace('.png', '.txt'))

        # Load grayscale image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype('float32') / 255.0
        img = torch.tensor(img).unsqueeze(0)  # [1, H, W] format

        # Load bounding boxes (YOLO format)
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    parts = list(map(float, line.strip().split()))
                    # Format: [class, x_center, y_center, width, height]
                    boxes.append(parts)
        boxes = torch.tensor(boxes, dtype=torch.float32)

        sample = {'image': img, 'labels': boxes}
        return sample
