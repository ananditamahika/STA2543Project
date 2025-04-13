# src/evaluate.py

import numpy as np

def compute_iou(box1, box2):
    """
    Computes IoU between two boxes in [x_center, y_center, w, h] format.
    """
    # Convert to [x1, y1, x2, y2]
    def convert(box):
        x, y, w, h = box
        return [x - w/2, y - h/2, x + w/2, y + h/2]

    box1 = convert(box1)
    box2 = convert(box2)

    # Intersection box
    x_left = max(box1[0], box2[0])
    y_top  = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    inter_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    iou = inter_area / (box1_area + box2_area - inter_area)
    return iou

def evaluate_predictions(pred_boxes_list, gt_boxes_list, iou_thresh=0.5):
    """
    Computes precision, recall, and IoU.

    Args:
        pred_boxes_list: List of lists of predicted boxes per image.
        gt_boxes_list: List of lists of ground truth boxes per image.
        iou_thresh: IoU threshold for a prediction to count as correct.

    Returns:
        precision, recall, mean_iou
    """
    total_gt = 0
    total_pred = 0
    correct = 0
    ious = []

    for preds, gts in zip(pred_boxes_list, gt_boxes_list):
        total_gt += len(gts)
        total_pred += len(preds)

        matched = set()
        for pred in preds:
            for i, gt in enumerate(gts):
                if i in matched:
                    continue
                iou = compute_iou(pred, gt)
                if iou >= iou_thresh:
                    correct += 1
                    ious.append(iou)
                    matched.add(i)
                    break

    precision = correct / total_pred if total_pred > 0 else 0
    recall = correct / total_gt if total_gt > 0 else 0
    mean_iou = np.mean(ious) if ious else 0

    return precision, recall, mean_iou
