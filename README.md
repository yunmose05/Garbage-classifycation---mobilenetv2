# Garbage Classification - MobileNetV2

基于 MobileNetV2 迁移学习的 6 类垃圾分类图像分类器。

## 效果展示
<img width="500" height="500" alt="test_result" src="https://github.com/user-attachments/assets/90e369f5-e4a9-4227-8da6-712d442dce54" />


## 分类类别

| 类别 | 英文 | 说明 |
|------|------|------|
| 纸板 | cardboard | 纸箱、硬纸板、纸盒 |
| 玻璃 | glass | 玻璃瓶、玻璃杯 |
| 金属 | metal | 易拉罐、金属罐 |
| 纸张 | paper | 报纸、杂志、纸袋 |
| 塑料 | plastic | 塑料瓶、塑料袋 |
| 其他垃圾 | trash | 电池、厨余、旧衣物等 |

## 快速开始

```bash
# 1. 激活环境
conda activate yolo11

# 2. 安装依赖
pip install torch torchvision matplotlib Pillow kagglehub scikit-learn imagehash tqdm

# 3. 下载数据集并训练
python stage1/download_s1.py          # 下载 TrashNet
python download_kaggle.py             # 下载 Kaggle 数据
python stage1/supplement_s1.py        # 补充数据
python stage1/clean_s1.py             # 清洗数据
python stage1/train_s1.py             # 训练（约 30-40 分钟）

# 4. 批量预测
# 将要测试的图片放入 test/ 文件夹
python batch_predict.py
# 结果在 result/ 文件夹中，汇总数据见 result/summary.csv
```

## 项目结构

```
├── download_kaggle.py      # Kaggle 数据下载 & 12→6 类映射
├── batch_predict.py        # 批量预测脚本
├── predict.py              # 单张预测脚本
├── stage1/
│   ├── download_s1.py      # TrashNet 下载
│   ├── supplement_s1.py    # 补充数据抽取
│   ├── clean_s1.py         # 数据清洗
│   └── train_s1.py         # 阶段一训练
├── stage2/                 # 阶段二增量训练
│   ├── prepare_s2.py
│   ├── clean_s2.py
│   └── train_s2.py
├── test/                   # 放入测试图片
└── result/                 # 预测结果输出
```

## 训练结果

10 epoch CPU 训练，验证集准确率最高 **84.1%**。

| Epoch | Loss | Train Acc | Val Acc |
|-------|------|-----------|---------|
| 1 | 0.99 | 0.644 | 0.734 |
| 5 | 0.49 | 0.831 | **0.841** |
| 10 | 0.45 | 0.849 | 0.839 |

## 数据集来源

- **TrashNet** ([garythung/trashnet](https://github.com/garythung/trashnet)) — 2,527 张，6 类
- **Garbage Classification 12 classes** ([Kaggle](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)) — 15,515 张，映射为 6 类

## License

MIT
