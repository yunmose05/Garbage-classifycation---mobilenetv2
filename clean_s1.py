"""阶段 1 数据清洗：损坏、低分辨率、重复（修复 Windows 文件锁问题）"""
from pathlib import Path
from PIL import Image
import imagehash
from collections import defaultdict

STAGE1 = Path(r"D:\Personal_work\杂\作业\人工智能\rubbish(deepseek)\mobilenet_project_v4\stage1")
DATA = STAGE1 / "dataset"
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

removed = {'corrupted': 0, 'lowres': 0, 'duplicate': 0}

for cls in CLASSES:
    cls_dir = DATA / cls
    if not cls_dir.exists():
        continue
    files = list(cls_dir.glob('*'))
    print(f"{cls}: 处理 {len(files)} 张...")

    valid = []
    for f in files:
        try:
            with Image.open(f) as img:
                img.verify()
            with Image.open(f) as img:
                w, h = img.size
            if w < 80 or h < 80:
                f.unlink(); removed['lowres'] += 1; continue
            valid.append(f)
        except Exception:
            try:
                f.unlink()
            except:
                pass
            removed['corrupted'] += 1

    hashes = defaultdict(list)
    for f in valid:
        try:
            with Image.open(f).convert('RGB') as img:
                h = str(imagehash.phash(img))
            hashes[h].append(f)
        except Exception:
            try:
                f.unlink()
            except:
                pass
            removed['corrupted'] += 1

    to_remove = []
    for h, dups in hashes.items():
        to_remove.extend(dups[1:])

    for f in to_remove:
        try:
            f.unlink()
            removed['duplicate'] += 1
        except Exception:
            pass

print(f"\n清洗完成：损坏 {removed['corrupted']} | 低分辨率 {removed['lowres']} | 重复 {removed['duplicate']}")
total = sum(len(list((DATA / c).glob('*'))) for c in CLASSES if (DATA / c).exists())
print(f"清洗后总计: {total} 张")
