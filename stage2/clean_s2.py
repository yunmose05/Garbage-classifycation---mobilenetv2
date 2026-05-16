"""阶段 2 数据清洗"""
from pathlib import Path
from PIL import Image
import imagehash
from collections import defaultdict

STAGE2 = Path(r"D:\Personal_work\杂\作业\人工智能\rubbish(deepseek)\mobilenet_project_v4\stage2")
DATA = STAGE2 / "dataset"
CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

removed = {'corrupted': 0, 'lowres': 0, 'duplicate': 0}
for cls in CLASSES:
    cls_dir = DATA / cls
    if not cls_dir.exists(): continue
    files = list(cls_dir.glob('*'))
    print(f"{cls}: {len(files)} 张...")
    valid = []
    for f in files:
        try:
            img = Image.open(f); img.verify()
            img = Image.open(f)
            if min(img.size) < 80:
                f.unlink(); removed['lowres'] += 1; continue
            valid.append(f)
        except Exception:
            f.unlink(); removed['corrupted'] += 1
    hashes = defaultdict(list)
    for f in valid:
        try:
            h = str(imagehash.phash(Image.open(f).convert('RGB')))
            hashes[h].append(f)
        except Exception:
            f.unlink(); removed['corrupted'] += 1
    for h, dups in hashes.items():
        for f in dups[1:]:
            f.unlink(); removed['duplicate'] += 1

print(f"\n清洗完成：损坏 {removed['corrupted']} | 低分辨率 {removed['lowres']} | 重复 {removed['duplicate']}")
total = sum(len(list((DATA / c).glob('*'))) for c in CLASSES if (DATA / c).exists())
print(f"清洗后: {total} 张")
