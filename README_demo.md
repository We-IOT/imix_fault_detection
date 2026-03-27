# LimiX 温振传感器故障检测代码详细解读

## 📖 目录

1. [代码概述](#代码概述)
2. [环境配置](#环境配置)
3. [代码逐节详解](#代码逐节详解)
4. [核心 API 说明](#核心 api 说明)
5. [运行指南](#运行指南)
6. [常见问题](#常见问题)

---

## 代码概述

本脚本 `demo_fault_detection.py` 参考 `demo_classification.py` 实现，展示了如何使用 LimiX 模型进行温振传感器故障检测。代码包含详细的中文注释，适合学习和理解 LimiX 的使用方式。

### 流程概览

```
1. 加载传感器数据 (CSV)
      ↓
2. 构建滑动窗口数据集 (窗口=30, 步长=10)
      ↓
3. 特征工程 (80 维特征：统计/趋势/异常/融合)
      ↓
4. 数据预处理 (MinMax 归一化 + 数据集划分)
      ↓
5. 初始化 LimiX-2M 模型 (cuda)
      ↓
6. 执行推理 (4 个估计器集成)
      ↓
7. 评估结果 (准确率、AUC、分类报告)
      ↓
8. 保存结果 (CSV + JSON)
```

---

## 环境配置

### 第 1 步：设置环境变量

```python
# 设置环境变量（LimiX 所需）
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"
```

**作用**：
- `RANK`: 进程排名（单进程设为 0）
- `WORLD_SIZE`: 进程总数（单进程设为 1）
- `MASTER_ADDR/PORT`: 主进程地址和端口

**为什么需要**：LimiX 基于分布式训练框架，即使单进程推理也需要这些环境变量。

### 第 2 步：添加项目路径

```python
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
```

**作用**：将项目根目录添加到 Python 路径，确保能正确导入 `inference.predictor` 等模块。

---

## 代码逐节详解

### 第 2 步：配置参数

```python
# 数据路径
DATA_PATH = './limix_fault_detection/data/raw/sensor_data.csv'

# 模型配置
MODEL_PATH = './LimiX-2M.ckpt'  # 使用 2M 模型（速度⚡，显存📉）
CONFIG_PATH = './config/cls_default_2M_retrieval.json'  # 带检索配置

# 窗口参数
WINDOW_SIZE = 30    # 滑动窗口大小（30 分钟）
SLIDE_STEP = 10     # 滑动步长（每 10 分钟采样一个样本）

# 数据划分比例
TEST_SIZE = 0.2     # 测试集比例 20%
RANDOM_STATE = 42   # 随机种子（确保可重复性）

# 归一化范围
NORM_RANGE = (0, 1) # MinMax 归一化到 [0, 1]
```

**参数说明**：

| 参数 | 值 | 作用 |
|-----|-|-|
| `MODEL_PATH` | `LimiX-2M.ckpt` | 轻量级模型（2M 参数），推理快 |
| `CONFIG_PATH` | `cls_default_2M_retrieval.json` | 带检索功能的配置 |
| `WINDOW_SIZE` | 30 | 每次取 30 个连续样本作为窗口 |
| `SLIDE_STEP` | 10 | 每次滑动 10 个样本 |

**为什么滑动窗口**：
- 温振传感器是时间序列数据
- 单个时间点的特征信息有限
- 窗口能捕捉趋势、变化率等时间依赖特征
- 步长 < 窗口 → 重叠窗口 → 更多样本

---

### 第 3 步：加载数据

```python
def load_data(data_path):
    # 读取 CSV 文件
    df = pd.read_csv(data_path)
    
    # 解析时间戳列
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 按时间排序
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # ... (打印信息)
```

**关键点**：
1. `pd.to_datetime()`: 将字符串转换为时间对象
2. `sort_values()`: 确保数据按时间顺序排列
3. `reset_index()`: 重新生成连续索引

**数据格式要求**：
- `timestamp`: 时间列（必需）
- `device_id`: 设备标识（可选）
- `fault_label`: 标签列（必需）
- 其他列为传感器读数

---

### 第 4 步：特征工程

```python
def prepare_features(df, window_size, slide_step):
    # 准备窗口数据集
    X, y, feature_names, label_encoder = prepare_windowed_dataset(
        df, 
        window_size=window_size, 
        slide_step=slide_step
    )
    return X, y, feature_names, label_encoder
```

**特征提取核心**：`prepare_windowed_dataset()` 函数

**实现流程**：
```python
# 伪代码解释
eventor = LowFreqFeatureExtractor()

for i in range(window_size, len(df), slide_step):
    window = df.iloc[i-window_size:i]  # 取窗口数据
    label = df.iloc[i]['fault_label']  # 取窗口最后一个样本的标签
    
    features = extractor.extract_features(window)
    # 对每个传感器列，提取:
    # - 统计特征：mean, std, min, max, range, median
    # - 趋势特征：polynomial 拟合斜率
    # - 变化率：(最后一个 - 第 一个)/第一个
    # - 异常特征：Z-score > 2 的样本数
    # 
    # 跨传感器特征:
    # - 温度 - 振动相关性
    # - 温度差、比率
    # - 振动总能量
```

**生成的 80 维特征**：

| 特征类型 | 示例 | 说明 |
|-------|-|-|
| 统计特征 | `temp_1_mean` | 窗口内温度 1 的均值 |
| 趋势特征 | `temp_1_trend` | 线性趋势斜率（正=上升，负=下降） |
| 变化率 | `temp_1_change_rate` | (temp_1[-1] - temp_1[0]) / temp_1[0] |
| 异常特征 | `vib_x_anomaly_count` | Z-score > 2.0 的异常点数量 |
| 融合特征 | `temp_vib_correlation` | 温度与振动的相关性 |
| 方向特征 | `vib_total` | √(vib_x² + vib_y² + vib_z²) |

**优势**：
- 捕捉时间序列趋势（趋势特征）
- 检测突变（变化率、异常计数）
- 识别多传感器关联（相关性、比率）

---

### 第 5 步：数据预处理

```python
def preprocess_data(X, y):
    # 归一化
    scaler = MinMaxScaler(feature_range=NORM_RANGE)
    X = scaler.fit_transform(X)
    
    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y  # 分层抽样
    )
    
    return X_train, X_test, y_train, y_test, scaler
```

**归一化**：
- `MinMaxScaler`: 将所有特征缩放到 [0, 1] 范围
- 公式：`x_normalized = (x - x_min) / (x_max - x_min)`
- 作用：消除量纲影响，加速模型收敛

**分层抽样**：
- `stratify=y`: 保持测试集标签分布与原始数据一致
- 避免测试集中某些类别样本过少

**示例**：
- 原始数据：正常 36%，轴承故障 19%，齿轮故障 19%...
- 训练集：36% : 19% : 19% ...
- 测试集：36% : 19% : 19% ...

---

### 第 6 步：初始化 LimiX 模型

```python
def init_model(model_path, config_path):
    # 检查 GPU 可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载配置文件
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 初始化 LimiX 预测器
    classifier = LimiXPredictor(
        device=device,
        model_path=model_path,
        inference_config=config,
        mix_precision=True  # 启用混合精度（加速推理）
    )
    
    return classifier, device
```

**LimiX 初始化参数**：

| 参数 | 类型 | 说明 |
|-----|-|---|
| `device` | `torch.device` | 推理设备（cuda/cpu） |
| `model_path` | `str` | 模型文件路径 |
| `inference_config` | `dict` 或 `str` | 推理配置（可以是 JSON 文件路径） |
| `mix_precision` | `bool` | 是否启用混合精度（fp16+bf16） |

**混合精度**：
- 使用 FP16 和部分 BF16 计算
- 加速推理 + 节省显存
- 在 NVIDIA Ampere 架构（RTX 30 系、40 系）上效果最佳

---

### 第 7 步：执行推理

```python
def run_inference(classifier, X_train, y_train, X_test):
    # LimiX 推理
    y_pred_proba = classifier.predict(
        X_train, y_train, X_test,
        task_type="Classification"
    )
    
    return y_pred_proba, inference_time
```

**LimiX 的 `predict` 方法**：

```python
# LimiXPredictor.predict() 核心逻辑（伪代码）

def predict(self, x_train, y_train, x_test, task_type):
    # 1. 验证数据
    x_train, y_train = self.validate_data(x_train, y_train)
    x_test = self.validate_data(x_test)
    
    # 2. 拼接训练集和测试集（确保预处理一致）
    x = np.concatenate([x_train, x_test], axis=0)
    
    # 3. 预处理
    x = self.convert_x_dtypes(x)           # 类型转换
    x = self.convert_category2num(x)       # 类别编码
    x = x.astype(np.float32)
    
    # 4. 集成推理（4 个估计器）
    outputs = []
    for pipe in self.preprocess_pipelines:
        # 每个估计器独立的预处理
        x_ = preprocess(x, pipe)
        
        # 模型推理
        output = self.model(x_, y_, task_type)
        outputs.append(output)
    
    # 5. 集成预测（平均）
    output = mean(stack(outputs))
    
    # 6. Softmax + 归一化
    output = softmax(output / temperature)
    
    return output
```

**为什么需要训练数据**：
- LimiX 是**少样本学习**模型，推理时参考训练数据
- 检索相似样本（config 中 `use_retrieval=True`）
- 训练数据用于特征注意力和样本注意力计算

---

### 第 8 步：评估结果

```python
def evaluate_performance(y_test, y_pred_proba, label_encoder):
    # 获取预测类别
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # 解码标签（数字 → 类别名）
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    
    # 计算指标
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    # AUC-ROC（多分类）
    auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovo')
```

**评估指标详解**：

| 指标 | 公式 | 说明 |
|-----|----|---|
| **Accuracy** | TP+TN / (TP+TN+FP+FN) | 预测正确的比例 |
| **F1-Macro** | 各类别 F1 的平均 | 平等对待所有类别 |
| **F1-Weighted** | 按样本数加权的 F1 平均 | 大类样本权重更高 |
| **AUC-ROC** | 曲线下面积 | 分类器性能（越高越好） |
| **OvO** | One-vs-One | 两两比较的 AUC |

**置信度分析**：
```python
confidences = np.max(y_pred_proba, axis=1)
"
平均置信度：0.9421
中位数置信度：0.9923
```

- 平均置信度高（0.94）→ 模型整体自信
- 中位数高（0.99）→ 大部分样本预测确定
- 某些类别置信度波动大 → 需要更多数据

---

### 第 9 步：保存结果

```python
def save_results(y_test_labels, y_pred_labels, y_pred_proba, label_encoder, save_dir='./results'):
    # 保存预测结果
    confidences = np.max(y_pred_proba, axis=1)
    results_df = pd.DataFrame({
        'true_label': y_test_labels,
        'pred_label': y_pred_labels,
        'confidence': confidences,
        'is_correct': y_test_labels == y_pred_labels
    })
    results_df.to_csv(os.path.join(save_dir, 'detection_results.csv'))
```

**输出文件**：
1. `detection_results.csv`: 每条预测的详细结果
2. `stats.json`: 统计摘要

---

## 核心 API 说明

### LimiXPredictor

```python
from inference.predictor import LimiXPredictor

clf = LimiXPredictor(
    device=torch.device('cuda'),          # 设备
    model_path='./LimiX-2M.ckpt',         # 模型路径
    inference_config=config_dict,          # 配置
    mix_precision=True                    # 混合精度
)
```

**关键方法**：

```python
# 分类任务
y_pred_proba = clf.predict(
    X_train, y_train, X_test, 
    task_type="Classification"
)
# 输出：(n_test, n_classes) 的概率矩阵

# 回归任务
y_pred = clf.predict(
    X_train, y_train, X_test,
    task_type="Regression"
)
# 输出：(n_test,) 的连续值
```

### 配置文件

```json
[
  {
    "retrieval_config": {
      "use_retrieval": true,
      "subsample_type": "sample",
      "retrieval_len": 16
    },
    "PolynomialInteractionGenerator": {...},
    "RebalanceFeatureDistribution": {...},
    "FeatureShuffler": {...}
  },
  ...
  // 总共 4 个估计器配置
]
```

**配置说明**：
- `use_retrieval`: 是否启用检索（true=使用相似历史样本）
- `subsample_type`: 采样方式（sample/feature）
- `PolynomialInteractionGenerator`: 多项式交互特征
- `RebalanceFeatureDistribution`: 特征分布平衡
- `FeatureShuffler`: 特征打乱（增强鲁棒性）

---

## 运行指南

### 完整流程

```bash
# 1. 生成模拟数据
cd LimiX
python limix_fault_detection/data/generate_sensor_data.py

# 2. 运行演示
python limix_fault_detection/demo_fault_detection.py
```

### 输出示例

```
=== Step 1: 加载传感器数据 ===
✓ 数据加载成功
  - 样本数量：1100 条
  - 标签分布：正常 400, 轴承故障 200, ...

=== Step 2: 特征工程（滑动窗口） ===
✓ 特征提取完成
  - 生成样本数：107
  - 特征维度：80

=== Step 3: 数据预处理 ===
✓ 数据集划分完成
  - 训练集：85 样本
  - 测试集：22 样本

=== Step 4: 初始化 LimiX 模型 ===
✓ 使用设备：cuda
✓ 模型配置加载

=== Step 5: 执行推理 ===
✓ 推理时间：1.90 秒

=== Step 6: 性能评估 ===
  准确率：1.0000
  F1 分数：1.0000
  AUC-ROC: 1.0000
```

---

## 常见问题

### Q1: 为什么需要训练数据才能推理？
**A**: LimiX 采用少样本学习范式，推理时会检索与测试样本相似的历史训练样本，利用其模式进行预测。这与传统分类器（仅学习决策边界）不同。

### Q2: 如何使用自定义数据？
**A**: 准备 CSV 文件，包含：
- `timestamp` 列
- 至少 1 个传感器列（`temp_x`, `vib_x` 等）
- `fault_label` 标签列
- 修改 `DATA_PATH` 指向你的文件

### Q3: 特征数量太多/太少怎么办？
**A**: 
- **太多** → 调整 `window_size` 减小窗口
- **太少** → 增大窗口或减小 `slide_step`
- 修改 `src/feature_extraction.py` 添加/删除特征

### Q4: 如何提高推理速度？
**A**: 
1. 使用 `LimiX-2M`（更小模型）
2. 修改 config 为 `cls_default_noretrieval.json`（禁用检索）
3. 增大 `batch_size`（批处理）

### Q5: 为什么有些类别置信度较低？
**A**: 可能是：
1. 训练样本太少
2. 类别特征不明显
3. 需要添加针对性特征

### Q6: 如何修改报警阈值？
**A**: 修改 `src/real_time_monitor.py` 中的 `confidence_threshold` 参数

---

## 总结

本代码展示了完整的 LimiX 故障检测流程：

| 阶段 | 关键代码 | 输出 |
|-----|-|-|
| 数据加载 | `pd.read_csv()` | DataFrame |
| 特征工程 | `prepare_windowed_dataset()` | (107, 80) |
| 预处理 | `MinMaxScaler + train_test_split()` | 训练/测试集 |
| 推理 | `LimiXPredictor.predict()` | 概率矩阵 |
| 评估 | `accuracy_score + f1_score` | 指标 |

**下一步**：
1. 收集真实传感器数据训练
2. 调整参数优化性能
3. 部署到生产环境
