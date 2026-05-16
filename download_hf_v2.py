"""
下载 HuggingFace 数据集（逐张从缓存加载，写入本地磁盘）
"""
from pathlib import Path
from PIL import Image
from io import BytesIO
from tqdm import tqdm

BASE = Path(r"D:\Personal_work\杂\作业\人工智能\rubbish(deepseek)\mobilenet_project_v4")
HF_CACHE = BASE / "hf_dataset" / "raw"

CLASSES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
CATEGORY_MAP = {
    'cardboard': 'cardboard', 'glass': 'glass', 'metal': 'metal',
    'paper': 'paper', 'plastic': 'plastic', 'trash': 'trash',
    'biological': 'trash', 'battery': 'trash', 'shoes': 'trash', 'clothes': 'trash',
}

for cls in CLASSES:
    (HF_CACHE / cls).mkdir(parents=True, exist_ok=True)

# 检查现有缓存，跳过已写入的文件
existing_counts = {}
for cls in CLASSES:
    existing_counts[cls] = len(list((HF_CACHE / cls).glob('*.jpg')))

total_existing = sum(existing_counts.values())
print(f"本地已有 {total_existing} 张图片")

from datasets import load_dataset
dataset = load_dataset("omasteam/waste-garbage-management-dataset", trust_remote_code=True)

counts = {cls: existing_counts[cls] for cls in CLASSES}
skipped = 0

for split_name, split_data in dataset.items():
    print(f"\n{'='*60}")
    print(f"Split: {split_name} ({len(split_data)} 条)")
    print(f"{'='*60}")

    for idx, item in enumerate(tqdm(split_data, desc=split_name)):
        try:
            raw_label = item['label']
            if isinstance(raw_label, int):
                raw_class = split_data.features['label'].int2str(raw_label)
            else:
                raw_class = str(raw_label)
            raw_lower = raw_class.lower().strip()

            target = None
            for key, val in CATEGORY_MAP.items():
                if key in raw_lower:
                    target = val; break
            if target is None:
                target = 'trash'

            image = item['image']
            if hasattr(image, 'save'):
                pil_img = image
            elif isinstance(image, dict):
                if 'bytes' in image:
                    pil_img = Image.open(BytesIO(image['bytes']))
                elif 'path' in image:
                    pil_img = Image.open(image['path'])
                else:
                    skipped += 1; continue
            elif isinstance(image, bytes):
                pil_img = Image.open(BytesIO(image))
            else:
                skipped += 1; continue

            cnt = counts[target]
            fname = f"{split_name}_{cnt:05d}.jpg"
            pil_img.convert('RGB').save(str(HF_CACHE / target / fname), 'JPEG', quality=95)
            counts[target] += 1

        except Exception as e:
            skipped += 1

print("\n" + "=" * 50)
print("下载完成！HF 数据集总计：")
total = sum(counts.values())
for cls in CLASSES:
    print(f"  {cls}: {counts[cls]}")
print(f"  总计: {total}")
print(f"  跳过: {skipped}")
