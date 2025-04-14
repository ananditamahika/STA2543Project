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
    import os
    import shutil
    import random

    print(f"\n[INFO] Preparing dataset split from:")
    print(f"       Images: {image_dir}")
    print(f"       Labels: {label_dir}")

    random.seed(seed)

    # 1. Get image files (only those with matching .txt labels)
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    matched_images = []
    for f in image_files:
        label_name = os.path.splitext(f)[0] + ".txt"
        if os.path.exists(os.path.join(label_dir, label_name)):
            matched_images.append(f)

    print(f"[INFO] Matched image-label pairs: {len(matched_images)}")

    if len(matched_images) == 0:
        print("[ERROR] No matched files found. Exiting.")
        return

    # 2. Shuffle + split
    random.shuffle(matched_images)
    val_count = int(len(matched_images) * val_ratio)

    splits = {
        "train": matched_images[val_count:],
        "val": matched_images[:val_count]
    }

    # 3. Save
    for split in splits:
        img_out = os.path.join(output_dir, split, "images")
        lbl_out = os.path.join(output_dir, split, "labels")
        os.makedirs(img_out, exist_ok=True)
        os.makedirs(lbl_out, exist_ok=True)

        for fname in splits[split]:
            shutil.copy(os.path.join(image_dir, fname), os.path.join(img_out, fname))
            label_file = os.path.splitext(fname)[0] + ".txt"
            shutil.copy(os.path.join(label_dir, label_file), os.path.join(lbl_out, label_file))

        print(f"[INFO] {split.capitalize()} set: {len(splits[split])} images")

def match_labels_by_basename(label_dir, image_dir, output_label_dir):
    """
    Match label files to image files by base filename and copy them to a new folder.

    Args:
        label_dir (str): Directory with original label .txt files.
        image_dir (str): Directory with extracted frames (image files).
        output_label_dir (str): Where to copy the matched .txt files.
    """
    os.makedirs(output_label_dir, exist_ok=True)

    image_basenames = {
        os.path.splitext(f)[0] for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".png"))
    }

    label_files = [
        f for f in os.listdir(label_dir)
        if f.lower().endswith(".txt") and os.path.splitext(f)[0] in image_basenames
    ]

    if not label_files:
        print(f"[WARN] No matched labels found in {label_dir}")
        return

    for label_file in label_files:
        src = os.path.join(label_dir, label_file)
        dst = os.path.join(output_label_dir, label_file)
        shutil.copy(src, dst)

    print(f"[INFO] Matched and copied {len(label_files)} label files to {output_label_dir}")
