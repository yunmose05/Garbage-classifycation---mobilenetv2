"""
阶段 1 训练：TrashNet + HF 补充，快速建立基线
用 test.jpg 做肉眼评估，不做复杂指标
"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Subset
import torch_directml
from pathlib import Path

STAGE1 = Path(r"D:\Personal_work\杂\作业\人工智能\rubbish(deepseek)\mobilenet_project_v4\stage1")
DATA_DIR = STAGE1 / "dataset"
BATCH_SIZE = 64
EPOCHS = 10
LR = 0.001
IMG_SIZE = 224

device = torch.device('cpu')
print(f"设备: {device} (DML 与 MobileNetV2 BatchNorm 不兼容，使用 CPU)")
BATCH_SIZE = 32

# 预处理
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_data = datasets.ImageFolder(str(DATA_DIR), transform=train_transform)
val_data = datasets.ImageFolder(str(DATA_DIR), transform=val_transform)
class_names = train_data.classes
print(f"类别: {class_names}")

n_total = len(train_data)
n_train = int(n_total * 0.8)
n_val = n_total - n_train
gen = torch.Generator().manual_seed(42)
train_idx, val_idx = random_split(range(n_total), [n_train, n_val], generator=gen)
train_set = Subset(train_data, train_idx.indices)
val_set = Subset(val_data, val_idx.indices)
train_loader = DataLoader(train_set, BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, BATCH_SIZE, shuffle=False)
print(f"训练: {n_train} | 验证: {n_val}")

model = models.mobilenet_v2(weights='IMAGENET1K_V1')
for p in model.features.parameters():
    p.requires_grad = False
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_acc = 0.0
print("\n===== 阶段 1 训练开始 =====\n")

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct = 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += (outputs.argmax(1) == labels).sum().item()

    model.eval()
    val_correct = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            val_correct += (model(imgs).argmax(1) == labels).sum().item()

    train_acc = correct / n_train
    val_acc = val_correct / n_val
    scheduler.step()

    marker = ""
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), str(STAGE1 / 'best_model_s1.pth'))
        marker = " [最优]"

    print(f"Epoch [{epoch+1:2d}/{EPOCHS}]  "
          f"Loss: {total_loss/len(train_loader):.4f}  "
          f"Train: {train_acc:.3f}  Val: {val_acc:.3f}{marker}")

print(f"\n===== 阶段 1 完成 =====")
print(f"最优验证准确率: {best_acc:.3f}")
print(f"模型: {STAGE1 / 'best_model_s1.pth'}")
