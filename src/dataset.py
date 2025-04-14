import os
import torch
from torch.utils.data import Dataset
import cv2

class YoloFishDataset(Dataset):
    def __init__(self, image_dir, label_dir, img_size=640):
        """
        Args:
            image_dir (str): Path to the folder with images.
            label_dir (str): Path to the folder with YOLO-format labels.
            img_size (int): Image size to resize to (square).
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.img_size = img_size

        # Get list of image files with matching labels
        all_images = os.listdir(image_dir)
        image_files = [f for f in all_images if f.lower().endswith(('.jpg', '.png'))]

        self.image_files = [
            f for f in image_files
            if os.path.isfile(os.path.join(label_dir, os.path.splitext(f)[0] + ".txt"))
        ]

        print(f"[INFO] YoloFishDataset initialized with {len(self.image_files)} image-label pairs")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + ".txt")

        # Load grayscale image and normalize
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"[ERROR] Could not load image: {img_path}")
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) / 255.0  # [1, H, W]
        image = image.repeat(3, 1, 1)

        # Load label in YOLO format: class x_center y_center width height
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    labels.append([float(p) for p in parts])

        # Add dummy if label is missing (optional, safe fallback)
        if len(labels) == 0:
            labels.append([0, 0, 0, 0, 0])

        labels = torch.tensor(labels, dtype=torch.float32)
        return {"image": image, "labels": labels}
