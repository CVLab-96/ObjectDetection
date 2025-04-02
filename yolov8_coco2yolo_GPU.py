import yaml
from ultralytics import YOLO
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path

# 计算损失
class ComplexCustomLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, iou_loss_weight=0.5):
        super().__init__()
        self.alpha = alpha  # For focal loss
        self.gamma = gamma  # For focal loss
        self.iou_loss_weight = iou_loss_weight  # Weight for IoU loss

    def forward(self, predictions, targets):
        # Separate predictions and targets for easier handling
        pred_cls, pred_boxes, pred_obj = predictions
        target_cls, target_boxes, target_obj = targets

        # Classification loss (CrossEntropy)
        cls_loss = F.cross_entropy(pred_cls, target_cls, reduction='mean')

        # Bounding box regression loss (Smooth L1 Loss)
        box_loss = F.smooth_l1_loss(pred_boxes, target_boxes, reduction='mean')

        # Objectness loss (BCE Loss)
        obj_loss = F.binary_cross_entropy_with_logits(pred_obj, target_obj, reduction='mean')

        # IoU Loss (to penalize poor bounding box overlap)
        iou_loss = self.iou_loss(pred_boxes, target_boxes)

        # Focal loss for class imbalance
        focal_loss = self.focal_loss(pred_cls, target_cls)

        # Combine all losses with respective weights
        total_loss = cls_loss + box_loss + obj_loss + (self.iou_loss_weight * iou_loss) + focal_loss
        return total_loss

    def iou_loss(self, pred_boxes, target_boxes):
        # Calculate IoU (Intersection over Union) loss
        inter_area = torch.minimum(pred_boxes[..., 2:], target_boxes[..., 2:]) - torch.maximum(pred_boxes[..., :2], target_boxes[..., :2])
        inter_area = torch.clamp(inter_area, min=0)
        intersection = inter_area[..., 0] * inter_area[..., 1]

        pred_area = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
        target_area = (target_boxes[..., 2] - target_boxes[..., 0]) * (target_boxes[..., 3] - target_boxes[..., 1])

        union_area = pred_area + target_area - intersection
        iou = intersection / (union_area + 1e-6)  # Add a small constant to avoid division by zero

        iou_loss = 1 - iou  # IoU loss is 1 - IoU (maximize IoU)
        return iou_loss.mean()

    def focal_loss(self, predictions, targets):
        # Focal Loss for class imbalance (from <url id="cvloa05nfo2s7l0v2hb0" type="url" status="parsed" title="" wc="50252">https://arxiv.org/abs/1708.02002</url> )
        pt = torch.exp(-F.cross_entropy(predictions, targets, reduction='none'))
        focal_loss = self.alpha * (1 - pt) ** self.gamma * F.cross_entropy(predictions, targets, reduction='none')
        return focal_loss.mean()

# 使用新的损失创建自定义 YOLO 模型类
class CustomYOLOWithComplexLoss(YOLO):
    def __init__(self, model_yaml):
        super().__init__(model_yaml)
        self.loss = ComplexCustomLoss()  # Use the complex custom loss

    def compute_loss(self, predictions, targets, model):
        # Override the compute_loss method to use custom loss
        return self.loss(predictions, targets)

if __name__ == "__main__":
    # 配置COCO数据集
    data_yaml = dict(
        train='./COCO2014/train/images',
        val='./COCO2014/val/images',
        test='./COCO2014/test/images',
        nc=80,
        names=[
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush'
        ]
    )

    # 将数据集配置写入data.yaml中
    with open('data.yaml', 'w') as outfile:
        yaml.dump(data_yaml, outfile, default_flow_style=False)

    # Training parameters
    optimizer_type = 'Adam'
    lr = 1e-4
    momentum = 0.9
    weight_decay = 5e-4

    scheduler_params = {
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
    }
    
    epochs = 10
    batch_size = 16
    imgsz = 640

    # 初始化YOLO模型
    model = CustomYOLOWithComplexLoss("yolov8x.yaml")

    # 配置参数并训练模型
    model.train(
        data="data.yaml",
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        optimizer=optimizer_type,
        lr0=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        warmup_epochs=scheduler_params['warmup_epochs'],
        warmup_momentum=scheduler_params['warmup_momentum'],
        warmup_bias_lr=scheduler_params['warmup_bias_lr']
    )

    # 显示训练结果
    results = model.val()
    print(results)