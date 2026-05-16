"""
下载 Kaggle Garbage Classification 数据集并映射到 6 类
保存到 hf_dataset/raw（与原有路径保持一致，方便后续脚本复用）
"""
import os, shutil, random
from pathlib import Path
import kagglehub

BASE = Path(r"D:\Personal_work\杂\作业\人工智能\rubbish(deepseek)\mobilenet_project_v4")
KAGGLE_CACHE = BASE / "hf_dataset" / "raw"

CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# 12 类 Kaggle → 6 类目标 映射规则
MAPPING = {
    'cardboard':    'cardboard',
    'glass':        'glass',
    'green-glass':  'glass',
    'brown-glass':  'glass',
    'white-glass':  'glass',
    'metal':        'metal',
    'paper':        'paper',
    'plastic':      'plastic',
    'trash':        'trash',
    'biological':   'trash',
    'battery':      'trash',
    'clothes':      'trash',
    'shoes':        'trash',
}

# 创建目标目录
for cls in CLASSES:
    (KAGGLE_CACHE / cls).mkdir(parents=True, exist_ok=True)

# 下载 Kaggle 数据集
print("正在从 Kaggle 下载数据集...")
kaggle_path = kagglehub.dataset_download('mostafaabla/garbage-classification')
print(f"Kaggle 数据集路径: {kaggle_path}")

source_dir = Path(kaggle_path) / "garbage_classification"
if not source_dir.exists():
    raise FileNotFoundError(f"未找到数据目录: {source_dir}")

print(f"\n原始类别统计：")
counts = {}
for d in source_dir.iterdir():
    if d.is_dir():
        cnt = len(list(d.glob('*')))
        counts[d.name] = cnt
        print(f"  {d.name}: {cnt}")

# 映射并复制
total_saved = {cls: 0 for cls in CLASSES}
for raw_cls in MAPPING:
    target_cls = MAPPING[raw_cls]
    src_dir = source_dir / raw_cls
    if not src_dir.exists():
        print(f"  [跳过] {raw_cls} 目录不存在")
        continue

    for f in src_dir.glob('*'):
        if f.suffix.lower() in ('.jpg', '.jpeg', '.png'):
            cnt = total_saved[target_cls]
            fname = f"kaggle_{cnt:05d}.jpg"
            shutil.copy2(str(f), str(KAGGLE_CACHE / target_cls / fname))
            total_saved[target_cls] += 1

print(f"\n映射后目标类别统计：")
total = 0
for cls in CLASSES:
    print(f"  {cls}: {total_saved[cls]}")
    total += total_saved[cls]
print(f"  总计: {total}")
print(f"\n数据集已保存到: {KAGGLE_CACHE}")
