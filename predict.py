"""
预测脚本：用最优权重对 test.jpg 做分类
优先使用阶段 2 权重，否则使用阶段 1
"""
import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(r"D:\Personal_work\杂\作业\人工智能\rubbish(deepseek)\mobilenet_project_v4")
IMG_SIZE = 224
class_names = ['cardboard (纸板)', 'glass (玻璃)', 'metal (金属)',
               'paper (纸张)', 'plastic (塑料)', 'trash (其他垃圾)']

# 检测最优可用权重
weights_path = None
stage = None
for s, p in [(2, BASE / "stage2" / "best_model_s2.pth"),
             (1, BASE / "stage1" / "best_model_s1.pth")]:
    if p.exists():
        weights_path = p
        stage = s
        break

if weights_path is None:
    raise FileNotFoundError("未找到训练好的模型，请先运行训练脚本")

print(f"使用阶段 {stage} 权重: {weights_path}")

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
model.load_state_dict(torch.load(str(weights_path), map_location='cpu', weights_only=True))
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img_path = BASE / 'test.jpg'
if not img_path.exists():
    print(f"请将测试图片放入: {img_path}")
    exit(1)

img = Image.open(str(img_path)).convert('RGB')
tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    outputs = model(tensor)
    probs = torch.softmax(outputs, dim=1)[0]
    pred = probs.argmax().item()

print("\n" + "=" * 40)
print(f"预测结果: {class_names[pred]}")
print(f"置信度:   {probs[pred]:.2%}")
print("-" * 40)
print("各类别概率:")
for i, (name, prob) in enumerate(zip(class_names, probs)):
    bar = '#' * int(prob * 40)
    print(f"  {name:20s}  {prob:.4f}  {bar}")
print("=" * 40)

plt.figure(figsize=(6, 6))
plt.imshow(img); plt.axis('off')
plt.title(f"预测: {class_names[pred]}\n置信度: {probs[pred]:.2%}", fontsize=14)
plt.tight_layout(); plt.savefig(str(BASE / 'result.jpg'), dpi=150)
print(f"\n结果图: {BASE / 'result.jpg'}")
