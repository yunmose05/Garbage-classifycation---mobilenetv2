"""
阶段 2 增量训练
- 加载阶段 1 的 best_model_s1.pth
- 在全新分布的数据上继续训练
- 学习率降低到 1e-4
- 解冻 MobileNetV2 最后 3 个 block 进行微调
"""
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split, Subset
import torch_directml
from pathlib import Path

BASE = Path(r"D:\Personal_work\杂\作业\人工智能\rubbish(deepseek)\mobilenet_project_v4")
STAGE1 = BASE / "stage1"
STAGE2 = BASE / "stage2"
DATA_DIR = STAGE2 / "dataset"
BATCH_SIZE = 64
EPOCHS = 12
LR_CLASSIFIER = 0.0001
LR_BACKBONE = 0.00001
UNFREEZE_LAST = 3
PATIENCE = 5
IMG_SIZE = 224

try:
    device = torch_directml.device()
except:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
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
assert class_names == ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
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

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))

# 加载阶段 1 权重
s1_path = STAGE1 / 'best_model_s1.pth'
if not s1_path.exists():
    raise FileNotFoundError(f"未找到阶段 1 权重: {s1_path}")
model.load_state_dict(torch.load(str(s1_path), map_location='cpu'))
print(f"已加载阶段 1 权重: {s1_path}")

# 解冻最后几个 block
features = list(model.features.children())
total_blocks = len(features)
freeze_until = total_blocks - UNFREEZE_LAST

for i, child in enumerate(features):
    for p in child.parameters():
        p.requires_grad = (i >= freeze_until)

for p in model.classifier.parameters():
    p.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
print(f"可训练参数: {trainable:,}/{total_params:,} ({trainable/total_params*100:.1f}%)")
print(f"解冻最后 {UNFREEZE_LAST}/{total_blocks} 个 block")

model = model.to(device)

optimizer = torch.optim.Adam([
    {'params': model.classifier.parameters(), 'lr': LR_CLASSIFIER},
    {'params': [p for i, child in enumerate(features)
                if i >= freeze_until for p in child.parameters()],
     'lr': LR_BACKBONE},
])

criterion = nn.CrossEntropyLoss()
sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

best_acc = 0.0
patience = 0
print(f"\n===== 阶段 2 增量训练 (基于阶段 1 权重) =====\n")

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
    sched.step()

    marker = ""
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), str(STAGE2 / 'best_model_s2.pth'))
        marker = " [最优]"
        patience = 0
    else:
        patience += 1

    print(f"Epoch [{epoch+1:2d}/{EPOCHS}]  "
          f"Loss: {total_loss/len(train_loader):.4f}  "
          f"Train: {train_acc:.3f}  Val: {val_acc:.3f}{marker}")

    if patience >= PATIENCE:
        print(f"\n早停！连续 {PATIENCE} 个 epoch 未提升。")
        break

print(f"\n===== 阶段 2 完成 =====")
print(f"最优验证准确率: {best_acc:.3f}")
print(f"模型: {STAGE2 / 'best_model_s2.pth'}")
