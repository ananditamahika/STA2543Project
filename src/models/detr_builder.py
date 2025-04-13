# src/models/detr_builder.py

from transformers import DetrForObjectDetection, DetrImageProcessor
import torch
import torch.optim as optim

def build_detr_model(num_classes=2, lr=1e-4):
    """
    Builds a DETR model using Hugging Face Transformers (pretrained on COCO),
    and modifies it for fine-tuning on 1-class (fish) + background.

    Args:
        num_classes (int): Number of object classes including background.
        lr (float): Learning rate.

    Returns:
        model, processor, optimizer
    """
    # Load pretrained DETR and replace the classification head
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )

    # Optional: freeze backbone if needed
    # for param in model.model.backbone.parameters():
    #     param.requires_grad = False

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    return model, processor, optimizer
