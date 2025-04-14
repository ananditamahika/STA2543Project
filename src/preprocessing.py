# src/preprocessing.py

import cv2
import os
import shutil
import random
from tqdm import tqdm

def extract_frames_from_video(video_path, output_dir, frame_rate=1):
    """
    Extracts grayscale frames from a video and saves them as PNGs.

    Args:
        video_path (str): Path to the input video file.
        output_dir (str): Path to store output frames.
        frame_rate (int): Extract 1 frame every `frame_rate` frames.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    frame_idx = 0
    saved_idx = 0

    print(f"[INFO] Extracting frames from {video_path}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_rate == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            filename = os.path.join(output_dir, f"{saved_idx:06d}.png")
            cv2.imwrite(filename, gray)
            saved_idx += 1

        frame_idx += 1

    cap.release()
    print(f"[INFO] Saved {saved_idx} grayscale frames to {output_dir}.")

def copy_labels(label_dir, frame_dir, output_label_dir):
    """
    Copies YOLO label files for frames that exist in frame_dir.

    Args:
        label_dir (str): Directory containing original label files.
        frame_dir (str): Directory with saved frame images.
        output_label_dir (str): Directory to copy matching label files.
    """
    os.makedirs(output_label_dir, exist_ok=True)
    for filename in os.listdir(frame_dir):
        base = os.path.splitext(filename)[0]
        label_file = os.path.join(label_dir, base + ".txt")
        if os.path.exists(label_file):
            shutil.copy(label_file, os.path.join(output_label_dir, base + ".txt"))

def split_dataset(image_dir, label_dir, output_dir, val_ratio=0.2, seed=42):
    """
    Splits image-label pairs into training and validation sets.

    Args:
        image_dir (str): Path to image directory.
        label_dir (str): Path to YOLO label directory.
        output_dir (str): Root output directory.
        val_ratio (float): Proportion of validation data.
    """
    import os
    import shutil
    import random

    random.seed(seed)
    image_files = [f for f in os.listdir(image_dir) if f.endswith(".png") or f.endswith(".jpg")]
    image_files = [f for f in image_files if os.path.exists(os.path.join(label_dir, os.path.splitext(f)[0] + ".txt"))]

    random.shuffle(image_files)

    val_count = int(len(image_files) * val_ratio)
    splits = {
        'train': image_files[val_count:],
        'val': image_files[:val_count]
    }

    for split in splits:
        img_out = os.path.join(output_dir, split, "images")
        lbl_out = os.path.join(output_dir, split, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for fname in splits[split]:
            shutil.copy(os.path.join(image_dir, fname), os.path.join(img_out, fname))
            label_fname = os.path.splitext(fname)[0] + ".txt"
            shutil.copy(os.path.join(label_dir, label_fname), os.path.join(lbl_out, label_fname))

        print(f"[INFO] {split.capitalize()} set: {len(splits[split])} images")
