"""
从 HF 缓存中提取 ~4000 张未被阶段 1 使用的图片
"""
import os, shutil
from pathlib import Path
import random

BASE = Path(r"D:\Personal_work\杂\作业\人工智能\rubbish(deepseek)\mobilenet_project_v4")
HF_CACHE = BASE / "hf_dataset" / "raw"
STAGE1 = BASE / "stage1"
STAGE2 = BASE / "stage2"
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

for cls in CLASSES:
    (STAGE2 / "dataset" / cls).mkdir(parents=True, exist_ok=True)

TOTAL_TARGET = 4000

# 获取阶段 1 已使用的 HF 文件名
used_names = set()
for cls in CLASSES:
    cls_dir = STAGE1 / "dataset" / cls
    if cls_dir.exists():
        for f in cls_dir.iterdir():
            if f.name.startswith('hf_supp_'):
                used_names.add(f.name)

print(f"阶段 1 使用了 {len(used_names)} 张 HF 图片")

# 从 HF 缓存中排除已用的，抽取 ~4000 张
copied = 0
for cls in CLASSES:
    hf_dir = HF_CACHE / cls
    available = [f for f in hf_dir.glob('*') if f.name not in used_names]

    if len(available) == 0:
        print(f"  {cls}: 无可抽取的图片")
        continue

    total_available = sum(1 for c in CLASSES
                          for f in (HF_CACHE / c).glob('*') if f.name not in used_names)
    proportion = len(available) / max(1, total_available)
    n = max(int(proportion * TOTAL_TARGET), 20)
    n = min(n, len(available))

    selected = random.sample(available, n) if n > 0 else []
    for f in selected:
        shutil.copy2(str(f), str(STAGE2 / "dataset" / cls / f.name))
    copied += len(selected)
    print(f"  {cls}: 抽取 {len(selected)} 张（可用: {len(available)}）")

print(f"\n阶段 2 数据集准备完成！共 {copied} 张")

# 输出阶段 2 统计
print("\n阶段 2 数据集统计：")
total = 0
for cls in CLASSES:
    cnt = len(list((STAGE2 / "dataset" / cls).glob('*')))
    total += cnt
    print(f"  {cls}: {cnt} 张")
print(f"  总计: {total} 张")
