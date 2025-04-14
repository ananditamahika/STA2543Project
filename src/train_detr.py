from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from transformers import DetrForObjectDetection, DetrImageProcessor
from dataset import YoloFishDataset
import os

def train_detr(
    train_img_dir,
    train_lbl_dir,
    val_img_dir,
    val_lbl_dir,
    epochs=10,
    save_path="weights/detr_fish.pth"
):
    print("[DEBUG] Loading datasets...")
    train_dataset = YoloFishDataset(train_img_dir, train_lbl_dir)
    val_dataset = YoloFishDataset(val_img_dir, val_lbl_dir)

    print(f"[DEBUG] Train dataset size: {len(train_dataset)}")
    print(f"[DEBUG] Val dataset size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2, collate_fn=lambda x: x)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=lambda x: x)

    print("[DEBUG] Loading pretrained DETR model...")
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=1,  # 1 class: fish
        ignore_mismatched_sizes=True
    )
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    for epoch in range(epochs):
        total_loss = 0.0
        print(f"\n🔁 Epoch {epoch + 1}/{epochs}")

        for batch in tqdm(train_loader):
            pixel_values = torch.stack([
                processor(images=sample["image"], return_tensors="pt")["pixel_values"][0]
                for sample in batch
            ]).to(device)

            target_labels = [
                {
                    "class_labels": sample["labels"][:, 0].long(),
                    "boxes": sample["labels"][:, 1:]
                }
                for sample in batch
            ]

            outputs = model(pixel_values=pixel_values, labels=target_labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[INFO] Epoch {epoch + 1} Loss: {total_loss:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f" Model saved to {save_path}")
