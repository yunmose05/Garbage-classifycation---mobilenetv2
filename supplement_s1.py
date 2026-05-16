"""
从 HF 缓存中随机抽取 ~2000 张图片补充到阶段 1 数据集
剩下的留给阶段 2 使用
"""
import os, shutil
from pathlib import Path
import random

BASE = Path(r"D:\Personal_work\杂\作业\人工智能\rubbish(deepseek)\mobilenet_project_v4")
HF_CACHE = BASE / "hf_dataset" / "raw"
STAGE1 = BASE / "stage1"
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

TARGET = STAGE1 / "dataset"

TOTAL_TARGET = 2000

hf_counts = {}
for cls in CLASSES:
    files = list((HF_CACHE / cls).glob('*'))
    hf_counts[cls] = files

total_available = sum(len(v) for v in hf_counts.values())

print("HF 缓存中各分类图片数：")
for cls in CLASSES:
    print(f"  {cls}: {len(hf_counts[cls])}")

print(f"\n从 HF 缓存中抽取约 {TOTAL_TARGET} 张补充到阶段 1...")

copied_total = 0
for cls in CLASSES:
    available = hf_counts[cls]
    proportion = len(available) / total_available if total_available > 0 else 0
    n = max(int(proportion * TOTAL_TARGET), 20)
    n = min(n, len(available))

    selected = random.sample(available, n) if n > 0 else []
    for f in selected:
        dst_name = f"hf_supp_{f.name}"
        shutil.copy2(str(f), str(TARGET / cls / dst_name))
    copied_total += len(selected)
    print(f"  {cls}: 补充 {len(selected)} 张")

print(f"\n阶段 1 补充完成！共补充 {copied_total} 张")

# 输出阶段 1 最终统计
print("\n阶段 1 数据集最终统计：")
total = 0
for cls in CLASSES:
    cnt = len(list((TARGET / cls).glob('*')))
    total += cnt
    print(f"  {cls}: {cnt} 张")
print(f"  总计: {total} 张")
