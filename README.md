# Garbage Classification - MobileNetV2

基于 MobileNetV2 迁移学习的 6 类垃圾分类图像分类器。

## 效果展示
<img width="500" height="500" alt="test_result" src="https://github.com/user-attachments/assets/90e369f5-e4a9-4227-8da6-712d442dce54" />
<img width="500" height="500" alt="paper_test_result" src="https://github.com/user-attachments/assets/739fab90-1b4b-435c-8563-acbb1a8adf18" />


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

## 训练过程数据

训练在 CPU 上执行，共 10 个 epoch，使用 4,223 张经过清洗的垃圾图片（8:2 划分为训练集与验证集）。结果如下：

| Epoch | Loss | Train Acc | Val Acc |
| --- | --- | --- | --- |
| 1   | 0.9900 | 0.644 | 0.734 |
| 2   | 0.6416 | 0.778 | 0.789 |
| 3   | 0.5635 | 0.801 | 0.801 |
| 4   | 0.5196 | 0.814 | 0.815 |
| 5   | 0.4946 | 0.831 | **0.841** |
| 6   | 0.4582 | 0.848 | 0.828 |
| 7   | 0.4664 | 0.834 | 0.827 |
| 8   | 0.4552 | 0.838 | 0.817 |
| 9   | 0.4379 | 0.848 | 0.825 |
| 10  | 0.4464 | 0.849 | 0.839 |

各指标含义：

- **Loss**：交叉熵损失，越低说明模型预测的概率分布越接近真实标签。
- **Train Acc**：训练集准确率，反映模型对训练数据的拟合程度。
- **Val Acc**：验证集准确率，反映模型对未见过数据的泛化能力，是评估模型好坏的主要指标。

### 训练结果分析

1. **迁移学习效果显著**：仅训练 10 个 epoch，最优验证准确率达到 **84.1%**，高于单数据集的基线水平（约 82%）。表明混合 TrashNet 和 Kaggle 数据的策略有效提升了模型的泛化能力。
2. **第 5 epoch 达到最优**：验证准确率在第 5 epoch 达到峰值 84.1%，之后略有波动，说明分类头在中小规模数据集上收敛较快，继续训练可能出现轻微过拟合。
3. **损失持续下降**：Loss 从第 1 个 epoch 的 0.99 稳步下降至第 10 个 epoch 的 0.45，降幅达 55%。

## 训练结果

10 epoch CPU 训练，验证集准确率最高 **84.1%**。

| Epoch | Loss | Train Acc | Val Acc |
|-------|------|-----------|---------|
| 1 | 0.99 | 0.644 | 0.734 |
| 5 | 0.49 | 0.831 | **0.841** |
| 10 | 0.45 | 0.849 | 0.839 |

## 批量预测结果

对 7 张测试图片进行批量预测的结果如下：

| 测试图片 | 预测类别 | 置信度 | 评估  |
| --- | --- | --- | --- |
| cardboard_test.jpg | cardboard | 63.08% | ✓ 正确 |
| glass_test.jpg | glass | 58.17% | ✓ 正确 |
| metal_test.jpg | **trash** | **89.09%** | ✗ **误分类** |
| paper_test.jpg | paper | 99.26% | ✓ 正确 |
| plastic_test.jpg | plastic | 50.39% | ✓ 正确 |
| test.jpg | cardboard | 98.65% | ✓ 正确 |
| trash_test.jpg | trash | 40.07% | ? 低置信度 |

分析：

- **model 对 paper（99.26%）和 cardboard（98.65%）的识别最为准确**，这两类特征明显、样本充足。
- **metal 被误分类为 trash（89.09%）**，与报告中金属和玻璃外观相近易混淆的结论一致。
- trash 类别置信度偏低（40.07%），因其类内差异大（包含电池、厨余、旧衣物等形态各异的物体），是后续优化方向。

## 数据集来源
- **TrashNet** ([garythung/trashnet](https://github.com/garythung/trashnet)) — 2,527 张，6 类
- **Garbage Classification 12 classes** ([Kaggle](https://www.kaggle.com/datasets/mostafaabla/garbage-classification)) — 15,515 张，映射为 6 类


