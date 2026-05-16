"""
阶段 1：下载 TrashNet（修正版）
TrashNet 的图片在 data/dataset-resized.zip 中，需要双重解压
"""
import os, shutil
from pathlib import Path

BASE = Path(r"D:\Personal_work\杂\作业\人工智能\rubbish(deepseek)\mobilenet_project_v4")
STAGE1 = BASE / "stage1"
DATASET = STAGE1 / "dataset"
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

for cls in CLASSES:
    (DATASET / cls).mkdir(parents=True, exist_ok=True)

print("正在下载 TrashNet...")
import urllib.request, zipfile

zip_path = STAGE1 / "trashnet.zip"
urllib.request.urlretrieve(
    "https://github.com/garythung/trashnet/archive/refs/heads/master.zip",
    str(zip_path)
)
print("解压中...")
with zipfile.ZipFile(str(zip_path), 'r') as zf:
    zf.extractall(str(STAGE1 / "tmp"))

tmp = STAGE1 / "tmp"

# TrashNet 结构: trashnet-master/data/dataset-resized.zip
dataset_zip = tmp / "trashnet-master" / "data" / "dataset-resized.zip"
if dataset_zip.exists():
    print("解压 dataset-resized.zip...")
    with zipfile.ZipFile(str(dataset_zip), 'r') as zf:
        zf.extractall(str(tmp / "trashnet-master" / "data"))

# 查找图片目录 - 可能在 dataset-resized/ 下
src = tmp / "trashnet-master" / "data" / "dataset-resized"
if not src.exists():
    # 也可能是直接在 data/ 下的子目录
    src = tmp / "trashnet-master" / "data"

print(f"数据源: {src}")

for cls in CLASSES:
    cls_src = None
    for d in src.iterdir():
        if d.is_dir() and d.name.lower() == cls.lower():
            cls_src = d
            break
    if cls_src is None:
        print(f"  [跳过] {cls} 目录未找到")
        continue
    count = 0
    for f in cls_src.iterdir():
        if f.suffix.lower() in ('.jpg', '.jpeg', '.png'):
            shutil.copy2(str(f), str(DATASET / cls / f.name))
            count += 1
    print(f"  {cls}: {count} 张")

# 清理
shutil.rmtree(str(tmp), ignore_errors=True)
zip_path.unlink(missing_ok=True)

print("\nTrashNet 下载完成！")
print("\n阶段 1 数据集统计（仅 TrashNet）：")
total = 0
for cls in CLASSES:
    cnt = len(list((DATASET / cls).glob('*')))
    total += cnt
    print(f"  {cls}: {cnt} 张")
print(f"  总计: {total} 张")
