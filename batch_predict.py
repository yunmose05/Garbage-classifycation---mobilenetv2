"""
批量预测脚本
读取 test/ 文件夹中的所有图片 -> 逐张分类 -> 保存到 result/ -> 输出 CSV 汇总
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
# 尝试使用 SimHei 字体，不存在则改用英文
try:
    fm.findfont('SimHei', fallback_to_default=False)
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
except Exception:
    matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(r"D:\Personal_work\杂\作业\人工智能\rubbish(deepseek)\mobilenet_project_v4")
TEST_DIR = BASE / "test"
RESULT_DIR = BASE / "result"
IMG_SIZE = 224
CLASS_NAMES = ['cardboard (纸板)', 'glass (玻璃)', 'metal (金属)',
               'paper (纸张)', 'plastic (塑料)', 'trash (其他垃圾)']

# 创建 result 目录
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ---- 检测最优权重 ----
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

print(f"使用阶段 {stage} 权重: {weights_path}\n")

# ---- 加载模型 ----
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, len(CLASS_NAMES))
model.load_state_dict(torch.load(str(weights_path), map_location='cpu', weights_only=True))
model.eval()

# ---- 图片预处理 ----
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ---- 收集测试图片 ----
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
test_files = sorted([
    f for f in TEST_DIR.iterdir()
    if f.is_file() and f.suffix.lower() in image_extensions
])

if not test_files:
    print(f"[错误] test/ 文件夹中没有找到图片（支持格式: jpg/png/bmp/tiff/webp）")
    print(f"请将测试图片放入: {TEST_DIR}")
    exit(1)

print(f"找到 {len(test_files)} 张测试图片\n")

# ---- 批量预测 ----
results = []

for i, img_path in enumerate(test_files, 1):
    print(f"[{i}/{len(test_files)}] 正在处理: {img_path.name}")

    try:
        img = Image.open(str(img_path)).convert('RGB')
    except Exception as e:
        print(f"  [跳过] 无法打开: {e}")
        continue

    # 推理
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred = probs.argmax().item()
        confidence = probs[pred].item()

    # 保存结果
    result_info = {
        'filename': img_path.name,
        'prediction': CLASS_NAMES[pred].split(' (')[0],
        'prediction_cn': CLASS_NAMES[pred],
        'confidence': confidence,
        'probs': {CLASS_NAMES[j].split(' (')[0]: probs[j].item() for j in range(len(CLASS_NAMES))},
    }
    results.append(result_info)

    # 输出到控制台
    pred_en = CLASS_NAMES[pred].split(' (')[0]
    print(f"  -> {pred_en} ({confidence:.2%})")

    # 保存可视化结果图
    result_name = f"{img_path.stem}_result{img_path.suffix}"
    result_path = RESULT_DIR / result_name

    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Prediction: {CLASS_NAMES[pred].split(' (')[0]}\nConfidence: {confidence:.2%}", fontsize=14)
    plt.tight_layout()
    plt.savefig(str(result_path), dpi=150)
    plt.close()

# ---- 控制台汇总 ----
print("\n" + "=" * 60)
print("批量预测完成！汇总：")
print("=" * 60)
correct_count = 0
for r in results:
    marker = "+" if r['confidence'] > 0.5 else "?"
    pred_en = r['prediction']
    print(f"  {marker} {r['filename']:25s} -> {pred_en:15s} ({r['confidence']:.2%})")
print("=" * 60)

# ---- 按类别统计 ----
from collections import Counter
pred_counts = Counter(r['prediction'] for r in results)
print("\n各类别预测数量：")
for name in sorted(pred_counts.keys()):
    print(f"  {name:12s}: {pred_counts[name]} 张")

# ---- 保存 CSV 报告 ----
csv_path = RESULT_DIR / "summary.csv"
with open(str(csv_path), 'w', encoding='utf-8-sig') as f:  # utf-8-sig 加 BOM, 防 Excel 乱码
    # 表头
    header = "文件名,预测类别,置信度"
    for cls_name in [c.split(' (')[0] for c in CLASS_NAMES]:
        header += f",{cls_name}"
    f.write(header + "\n")

    # 数据行
    for r in results:
        line = f"{r['filename']},{r['prediction']},{r['confidence']:.4f}"
        for cls_name in [c.split(' (')[0] for c in CLASS_NAMES]:
            line += f",{r['probs'][cls_name]:.4f}"
        f.write(line + "\n")

print(f"\n详细报告已保存: {csv_path}")
print(f"结果图片目录: {RESULT_DIR}")
print("\n各图片检测结果：")
for r in results:
    print(f"  {r['filename']:25s} {r['prediction']:15s} 置信度: {r['confidence']:.2%}")
