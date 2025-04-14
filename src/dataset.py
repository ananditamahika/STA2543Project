# src/dataset.py

import os
import torch
from torch.utils.data import Dataset
import cv2

class YoloFishDataset(Dataset):
    def __init__(self, image_dir, label_dir, img_size=640):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.img_size = img_size

        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        # Only keep images that have matching labels
        self.image_files = [
            f for f in self.image_files
            if os.path.isfile(os.path.join(label_dir, os.path.splitext(f)[0] + ".txt"))
        ]

        print(f"[INFO] YoloFishDataset initialized with {len(self.image_files)} image-label pairs")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + '.txt')

        # Load and preprocess image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0  # [1, H, W]

        # Load label
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    labels.append([float(p) for p in parts])
        labels = torch.tensor(labels, dtype=torch.float32)

        return {"image": image, "labels": labels}
