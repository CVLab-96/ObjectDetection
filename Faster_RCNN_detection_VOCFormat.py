import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os
import xml.etree.ElementTree as ET
import random
import torch.optim as optim

# 配置参数
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
NUM_CLASSES = len(VOC_CLASSES) + 1  # 20类 + 背景
BATCH_SIZE = 2
NUM_EPOCHS = 30
LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0005


# 自定义数据集类
class CustomVOCDataset(Dataset):
    def __init__(self, root_dir, image_set='train', transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_ids = []

        # 自动检测可能的划分文件
        possible_files = [
            os.path.join(root_dir, 'ImageSets', 'Main', f'{image_set}.txt'),
            os.path.join(root_dir, 'ImageSets', 'Main', f'{image_set}val.txt'),
            os.path.join(root_dir, 'ImageSets', 'Main', f'{image_set}_val.txt')
        ]

        for file_path in possible_files:
            if os.path.exists(file_path):
                with open(file_path) as f:
                    self.image_ids = [line.strip() for line in f.readlines()]
                break
        else:
            raise FileNotFoundError(f"No valid split file found for {image_set}")

        self.image_dir = os.path.join(root_dir, 'JPEGImages')
        self.annotation_dir = os.path.join(root_dir, 'Annotations')

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        image = Image.open(image_path).convert("RGB")

        # 解析XML标注
        xml_path = os.path.join(self.annotation_dir, f"{image_id}.xml")
        tree = ET.parse(xml_path)
        root = tree.getroot()

        boxes = []
        labels = []
        for obj in root.iter('object'):
            cls_name = obj.find('name').text
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(VOC_CLASSES.index(cls_name) + 1)  # 背景类为0

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        if self.transform:
            image = self.transform(image)

        return image, target, image_path


# 数据增强
def get_transform(train):
    transforms = []
    transforms.append(torchvision.transforms.ToTensor())
    if train:
        transforms.append(torchvision.transforms.RandomHorizontalFlip(0.5))
    return torchvision.transforms.Compose(transforms)


# 数据加载器
def collate_fn(batch):
    images = []
    targets = []
    paths = []
    for item in batch:
        images.append(item[0])
        targets.append(item[1])
        paths.append(item[2])
    return images, targets, paths


# 创建数据集
train_dataset = CustomVOCDataset(
    root_dir='./VOC2007',
    image_set='train',
    transform=get_transform(train=True)
)

val_dataset = CustomVOCDataset(
    root_dir='./VOC2007',
    image_set='val',
    transform=get_transform(train=False)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    collate_fn=collate_fn,
    shuffle=False
)


# 模型定义
def create_model():
    backbone = torchvision.models.mobilenet_v2(pretrained=True).features
    backbone.out_channels = 1280

    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),
        aspect_ratios=((0.5, 1.0, 2.0),)
    )

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0'],
        output_size=7,
        sampling_ratio=2
    )

    model = FasterRCNN(
        backbone,
        num_classes=NUM_CLASSES,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    return model


model = create_model().to(DEVICE)
optimizer = optim.SGD(
    [p for p in model.parameters() if p.requires_grad],
    lr=LEARNING_RATE,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY
)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


# 训练函数
def train():
    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        for batch_idx, (images, targets, _) in enumerate(train_loader):
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()
            if batch_idx % 50 == 0:
                print(f'Epoch: {epoch} | Batch: {batch_idx} | Loss: {losses.item():.4f}')

        lr_scheduler.step()
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch} Average Loss: {avg_loss:.4f}')
        torch.save(model.state_dict(), f'fasterrcnn_epoch.pth')


# 可视化对比函数
def visualize_comparison(image_path, prediction, target, confidence_threshold=0.5):
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    # 绘制真实标注（绿色）
    for box, label in zip(target['boxes'].cpu(), target['labels'].cpu()):
        box = box.numpy().astype(int)
        draw.rectangle([(box[0], box[1]), (box[2], box[3])],
                       outline="green", width=2)
        draw.text((box[0], box[1]), f"GT: {VOC_CLASSES[label - 1]}", fill="green")

    # 绘制预测结果（红色）
    for box, label, score in zip(prediction['boxes'].cpu(),
                                 prediction['labels'].cpu(),
                                 prediction['scores'].cpu()):
        if score > confidence_threshold:
            box = box.numpy().astype(int)
            draw.rectangle([(box[0], box[1]), (box[2], box[3])],
                           outline="red", width=2)
            text = f"Pred: {VOC_CLASSES[label - 1]} ({score:.2f})"
            draw.text((box[0], box[1] + 20), text, fill="red")

    plt.figure(figsize=(15, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.show()


# 验证可视化
def visualize_validation(model, dataloader, num_samples=3):
    model.eval()
    random_indices = random.sample(range(len(dataloader)), num_samples)

    for idx in random_indices:
        image, target, path = dataloader.dataset[idx]
        print(image)
        print(target)
        print(path)
        image_tensor = image.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            prediction = model(image_tensor)[0]

        print(f"Visualizing: {path}")
        visualize_comparison(path, prediction, target)


# 主程序
if __name__ == "__main__":
    # 训练模型
    train()

    # 加载最佳模型
    model.load_state_dict(torch.load('fasterrcnn_epoch.pth'))

    # 可视化验证集
    visualize_validation(model, val_loader, num_samples=5)
