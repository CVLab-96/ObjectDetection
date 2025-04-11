import os
import torch
import torchvision
from torchvision.models.detection import SSD
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.transforms import functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random
import xml.etree.ElementTree as ET
from PIL import Image

# 配置参数（可根据需要调整）
CONFIG = {
    "data_root": "./yolo2voc",          # 数据集根目录
    "classes": [                        # 必须与标注文件中的类别完全一致
        '0', '1', '2', '3', '4', '5', '6',
        '7', '8', '9', '10', '11', '12', '13',
        '14', '15', '16', '17'
    ],
    "batch_size": 4,                    # 根据GPU显存调整（最低2）
    "num_epochs": 20,
    "lr": 1e-4,                        # 降低学习率防止NaN
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "image_size": (300, 300),
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 2,
    "visual_num": 5,                    # 可视化数量
    "confidence_threshold": 0.5,
    "grad_clip": 10.0                   # 梯度裁剪阈值
}


class VOCDataset(Dataset):
    def __init__(self, root, image_set, transform=None):
        self.root = root
        self.transform = transform

        # 读取划分文件
        split_file = os.path.join(root, 'ImageSets', 'Main', f'{image_set}.txt')
        with open(split_file, 'r') as f:
            self.ids = [line.strip() for line in f if line.strip()]

        # 验证基础文件结构
        self._validate_structure()
        # 验证样本文件存在性
        self._validate_samples()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        while True:  # 跳过无效样本的循环
            try:
                img_id = self.ids[idx]

                # 加载图像
                img_path = os.path.join(self.root, 'JPEGImages', f'{img_id}.jpg')
                image = Image.open(img_path).convert('RGB')

                # 加载标注
                anno_path = os.path.join(self.root, 'Annotations', f'{img_id}.xml')
                target = self._parse_annotation(anno_path)

                if self.transform:
                    image, target = self.transform(image, target)

                # 检查空标注
                if len(target['boxes']) == 0:
                    raise ValueError("Empty annotation")

                return image, target
            except Exception as e:
                print(f"跳过无效样本 {img_id}: {str(e)}")
                idx = random.choice(range(len(self)))  # 随机选择新索引

    def _validate_structure(self):
        """验证数据集目录结构"""
        required_dirs = ['Annotations', 'JPEGImages', 'ImageSets/Main']
        for d in required_dirs:
            if not os.path.exists(os.path.join(self.root, d)):
                raise FileNotFoundError(f"缺少必要目录: {d}")

    def _validate_samples(self):
        """验证样本文件存在性"""
        missing = []
        for img_id in self.ids[:100]:  # 抽样检查前100个样本
            img_path = os.path.join(self.root, 'JPEGImages', f'{img_id}.jpg')
            anno_path = os.path.join(self.root, 'Annotations', f'{img_id}.xml')
            if not os.path.exists(img_path):
                missing.append(img_path)
            if not os.path.exists(anno_path):
                missing.append(anno_path)
        if missing:
            raise FileNotFoundError(f"缺失 {len(missing)} 个文件，示例：{missing[:3]}")

    def _parse_annotation(self, path):
        """安全的XML解析方法"""
        try:
            tree = ET.parse(path)
            root = tree.getroot()

            # 解析图像尺寸
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)

            objects = []
            for obj in root.iter('object'):
                name = obj.find('name').text
                if name not in CONFIG['classes']:
                    continue

                bndbox = obj.find('bndbox')
                xmin = max(0.0, float(bndbox.find('xmin').text))
                ymin = max(0.0, float(bndbox.find('ymin').text))
                xmax = min(width, float(bndbox.find('xmax').text))
                ymax = min(height, float(bndbox.find('ymax').text))

                # 过滤无效框
                if xmin >= xmax or ymin >= ymax:
                    continue

                objects.append({
                    'name': name,
                    'bbox': [xmin, ymin, xmax, ymax]
                })

            return {
                'size': (width, height),
                'objects': objects
            }
        except Exception as e:
            raise ValueError(f"解析 {path} 失败: {str(e)}")


class VOCTransform:
    def __init__(self, size=(300, 300)):
        self.size = size

    def __call__(self, image, target):
        # 图像预处理
        image = F.resize(image, self.size)
        image = F.to_tensor(image)
        image = F.normalize(image, mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])

        # 标注转换
        orig_w, orig_h = target['size']
        boxes = []
        labels = []

        for obj in target['objects']:
            xmin, ymin, xmax, ymax = obj['bbox']

            # 坐标转换
            scale_x = self.size[0] / orig_w
            scale_y = self.size[1] / orig_h
            xmin = xmin * scale_x
            ymin = ymin * scale_y
            xmax = xmax * scale_x
            ymax = ymax * scale_y

            # 最终边界检查
            xmin = max(0, min(xmin, self.size[0] - 1))
            ymin = max(0, min(ymin, self.size[1] - 1))
            xmax = max(xmin + 1, min(xmax, self.size[0]))
            ymax = max(ymin + 1, min(ymax, self.size[1]))

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(CONFIG['classes'].index(obj['name']))

        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64)
        }
        return image, target


def create_model(num_classes):
    # 初始化模型（加载预训练骨干网络）
    model = torchvision.models.detection.ssd300_vgg16(
        num_classes=num_classes,
        pretrained_backbone=True
    )
    # 权重初始化
    for param in model.backbone.parameters():
        param.requires_grad = True
    return model


def collate_fn(batch):
    return tuple(zip(*batch))


def train():
    # 初始化数据集
    train_dataset = VOCDataset(
        root=CONFIG['data_root'],
        image_set='train',
        transform=VOCTransform(CONFIG['image_size'])
    )
    val_dataset = VOCDataset(
        root=CONFIG['data_root'],
        image_set='val',
        transform=VOCTransform(CONFIG['image_size'])
    )

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        collate_fn=collate_fn
    )

    # 初始化模型
    model = create_model(num_classes=len(CONFIG['classes']) + 1)
    model.to(CONFIG['device'])

    # 优化器配置
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=CONFIG['lr'],
        momentum=CONFIG['momentum'],
        weight_decay=CONFIG['weight_decay']
    )

    # 训练循环
    best_loss = float('inf')
    for epoch in range(CONFIG['num_epochs']):
        model.train()
        total_loss = 0.0
        valid_samples = 0

        with tqdm(train_loader, unit="batch") as pbar:
            pbar.set_description(f"Epoch {epoch + 1}/{CONFIG['num_epochs']}")

            for images, targets in pbar:
                try:
                    # 数据转移
                    images = [img.to(CONFIG['device']) for img in images]
                    targets = [{k: v.to(CONFIG['device']) for k, v in t.items()}
                               for t in targets]

                    # 前向传播
                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                    # NaN检测
                    if torch.isnan(losses):
                        print("检测到NaN损失，跳过该批次")
                        continue

                    # 反向传播
                    optimizer.zero_grad()
                    losses.backward()

                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=CONFIG['grad_clip']
                    )

                    optimizer.step()

                    # 统计
                    total_loss += losses.item()
                    valid_samples += 1
                    pbar.set_postfix(loss=losses.item())

                except Exception as e:
                    print(f"批次处理失败: {str(e)}")
                    continue

        # 验证阶段
        model.train()  # 验证时计算损失需要模型处于训练模式
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(CONFIG['device']) for img in images]
                targets = [{k: v.to(CONFIG['device']) for k, v in t.items()}
                           for t in targets]

                loss_dict = model(images, targets)
                val_loss += sum(loss.item() for loss in loss_dict.values())

        model.eval()  # 切回评估模式

        # 计算平均损失
        avg_train_loss = total_loss / valid_samples if valid_samples > 0 else float('inf')
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # 保存最佳模型
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save(model.state_dict(), 'ssd_best.pth')
            print("保存新最佳模型")

    return model


def visualize():
    # 加载最佳模型
    model = create_model(num_classes=len(CONFIG['classes']) + 1)
    model.load_state_dict(torch.load('ssd_best.pth'))
    model.to(CONFIG['device'])
    model.eval()

    # 初始化测试数据集
    test_dataset = VOCDataset(
        root=CONFIG['data_root'],
        image_set='test',
        transform=VOCTransform(CONFIG['image_size'])
    )

    # 可视化函数
    def denormalize(image):
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        return image * std[:, None, None] + mean[:, None, None]

    # 随机选择样本
    indices = random.sample(range(len(test_dataset)), CONFIG['visual_num'])

    for idx in indices:
        image, target = test_dataset[idx]
        img_tensor = image.unsqueeze(0).to(CONFIG['device'])

        with torch.no_grad():
            prediction = model(img_tensor)[0]

        # 过滤预测结果
        mask = prediction['scores'] > CONFIG['confidence_threshold']
        boxes = prediction['boxes'][mask].cpu().numpy()
        labels = prediction['labels'][mask].cpu().numpy()
        scores = prediction['scores'][mask].cpu().numpy()

        # 准备图像
        image = denormalize(image).permute(1, 2, 0).cpu().numpy()
        image = np.clip(image, 0, 1)

        # 绘制结果
        plt.figure(figsize=(12, 8))
        ax = plt.gca()
        ax.imshow(image)

        for box, label, score in zip(boxes, labels, scores):
            xmin, ymin, xmax, ymax = box
            rect = plt.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                linewidth=1, edgecolor='lime', facecolor='none'
            )
            ax.add_patch(rect)
            plt.text(
                xmin, ymin - 5,
                f"{CONFIG['classes'][label]}: {score:.2f}",
                color='white',
                bbox=dict(facecolor='green', alpha=0.7, pad=1),
                fontsize=9,
                weight='bold'
            )

        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    # 检查CUDA可用性
    if CONFIG['device'] == 'cuda' and not torch.cuda.is_available():
        print("警告：CUDA不可用，将使用CPU")
        CONFIG['device'] = 'cpu'

    # 训练模型
    trained_model = train()

    # 可视化测试结果
    visualize()
