#!/usr/bin/env python
"""
温振传感器故障检测 - LimiX 模型演示

本脚本参考 demo_classification.py，针对温振传感器数据集实现完整的故障检测流程。
包含详细注释，适合学习和理解 LimiX 模型的使用。

主要步骤：
1. 加载传感器数据
2. 构建滑动窗口数据集
3. 特征工程（80 维特征提取）
4. 数据预处理（归一化、数据集划分）
5. 初始化 LimiX 模型
6. 执行推理
7. 评估结果（准确率、AUC 等指标）
8. 可视化分析（分类报告、混淆矩阵）
"""

import os
import sys
import time
import json
import warnings

# 导入必要的库
import numpy as np
import pandas as pd
import torch

# 数据预处理库
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)

# 忽略警告
warnings.filterwarnings('ignore')

# ============================================
# 第 1 步：设置环境
# ============================================

# 设置环境变量（LimiX 所需）
os.environ["RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = "29500"

# 添加项目根目录到路径
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# 导入 LimiX 预测器和特征提取器
from inference.predictor import LimiXPredictor
from src.feature_extraction import prepare_windowed_dataset, LowFreqFeatureExtractor

# ============================================
# 第 2 步：配置参数
# ============================================

# 数据路径
DATA_PATH = './limix_fault_detection/data/raw/sensor_data.csv'

# 模型配置
MODEL_PATH = './LimiX-2M.ckpt'  # 使用 2M 模型（速度快，显存占用低）
CONFIG_PATH = './config/cls_default_2M_retrieval.json'  # 带检索配置

# 窗口参数
WINDOW_SIZE = 30    # 滑动窗口大小（30 分钟）
SLIDE_STEP = 10     # 滑动步长（每 10 分钟采样一个样本）

# 数据划分比例
TEST_SIZE = 0.2     # 测试集比例 20%
RANDOM_STATE = 42   # 随机种子

# 归一化范围
NORM_RANGE = (0, 1) # MinMax 归一化到 [0, 1]

# ============================================
# 第 3 步：加载数据
# ============================================

def load_data(data_path):
    """
    加载传感器 CSV 数据
    
    Args:
        data_path: CSV 文件路径
        
    Returns:
        df: pandas DataFrame
    """
    print("=" * 80)
    print("Step 1: 加载传感器数据")
    print("=" * 80)
    
    # 读取 CSV 文件
    df = pd.read_csv(data_path)
    
    # 解析时间戳列
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 按时间排序
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"✓ 数据加载成功")
    print(f"  - 数据路径：{data_path}")
    print(f"  - 样本数量：{len(df)} 条")
    print(f"  - 特征列数：{df.shape[1]} 列")
    print(f"  - 时间范围：{df['timestamp'].min()} 到 {df['timestamp'].max()}")
    
    # 显示标签分布
    print(f"\n  - 标签分布:")
    label_counts = df['fault_label'].value_counts()
    for label, count in label_counts.items():
        print(f"    {label:12s}: {count:4d} 条 ({count/len(df)*100:.1f}%)")
    
    return df

# ============================================
# 第 4 步：特征工程
# ============================================

def prepare_features(df, window_size, slide_step):
    """
    使用滑动窗口提取特征
    
    滑动窗口说明:
    - 窗口大小：30（每次使用 30 个连续样本）
    - 步长：10（每次滑动 10 个样本）
    - 每个窗口生成 1 个特征向量和对应的标签
    
    Args:
        df: 原始数据 DataFrame
        window_size: 窗口大小
        slide_step: 滑动步长
        
    Returns:
        X: 特征矩阵 (n_samples, n_features)
        y: 标签数组
        feature_names: 特征名列表
        label_encoder: 标签编码器
    """
    print("\n" + "=" * 80)
    print("Step 2: 特征工程（滑动窗口）")
    print("=" * 80)
    
    # 准备窗口数据集
    X, y, feature_names, label_encoder = prepare_windowed_dataset(
        df, 
        window_size=window_size, 
        slide_step=slide_step
    )
    
    print(f"\n✓ 特征提取完成")
    print(f"  - 窗口大小：{window_size}")
    print(f"  - 滑动步长：{slide_step}")
    print(f"  - 生成样本数：{X.shape[0]}")
    print(f"  - 特征维度：{X.shape[1]}")
    
    # 显示特征类别
    feature_types = {
        '统计特征': [f for f in feature_names if '_mean' in f or '_std' in f],
        '趋势特征': [f for f in feature_names if 'trend' in f],
        '异常特征': [f for f in feature_names if 'anomaly' in f or 'zscore' in f],
        '融合特征': [f for f in feature_names if 'correlation' in f or 'ratio' in f]
    }
    
    print(f"\n  - 特征类型示例:")
    for ftype, feats in feature_types.items():
        if feats:
            print(f"    {ftype:10s}: {feats[0]:30s} ...")
    
    return X, y, feature_names, label_encoder

# ============================================
# 第 5 步：数据预处理
# ============================================

def preprocess_data(X, y):
    """
    数据预处理：归一化 + 数据集划分
    
    预处理流程:
    1. MinMax 归一化：将所有特征缩放到 [0, 1] 范围
    2. 数据集划分：训练集 80%，测试集 20%
    3. 分层抽样：保持测试集标签分布与原始数据一致
    
    Args:
        X: 特征矩阵
        y: 标签数组
        
    Returns:
        X_train, X_test, y_train, y_test, scaler: 预处理后的数据
    """
    print("\n" + "=" * 80)
    print("Step 3: 数据预处理")
    print("=" * 80)
    
    # 归一化
    scaler = MinMaxScaler(feature_range=NORM_RANGE)
    X = scaler.fit_transform(X)
    print(f"✓ 数据归一化完成 (范围：{NORM_RANGE})")
    
    # 数据集划分
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y  # 分层抽样
    )
    
    print(f"✓ 数据集划分完成")
    print(f"  - 训练集：{X_train.shape[0]} 样本")
    print(f"  - 测试集：{X_test.shape[0]} 样本")
    print(f"  - 划分比例：{len(X_train)/(len(X_train)+len(X_test))*100:.0f}% : {len(X_test)/(len(X_train)+len(X_test))*100:.0f}%")
    
    return X_train, X_test, y_train, y_test, scaler

# ============================================
# 第 6 步：初始化 LimiX 模型
# ============================================

def init_model(model_path, config_path):
    """
    初始化 LimiX 模型
    
    LimiX 模型说明:
    - LimiX-2M：轻量级模型（2M 参数），推理快，显存占用低
    - LimiX-16M：全尺寸模型（16M 参数），精度更高
    
    配置说明:
    - cls_default_2M_retrieval.json: 带检索功能的配置
    - cls_default_noretrieval.json: 无检索配置（更快）
    
    Args:
        model_path: 模型文件路径
        config_path: 配置文件路径
        
    Returns:
        classifier: LimiX 分类器实例
        device: 使用的设备（GPU/CPU）
    """
    print("\n" + "=" * 80)
    print("Step 4: 初始化 LimiX 模型")
    print("=" * 80)
    
    # 检查 GPU 可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ PyTorch 版本：{torch.__version__}")
    print(f"✓ 使用设备：{device}")
    
    if device.type == 'cuda':
        print(f"  - GPU 名称：{torch.cuda.get_device_name(0)}")
        print(f"  - GPU 显存：{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 加载配置文件
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"\n✓ 模型配置加载")
    print(f"  - 模型文件：{model_path}")
    print(f"  - 配置文件：{config_path}")
    print(f"  - 估计器数量：{len(config)}")
    
    # 初始化 LimiX 预测器
    classifier = LimiXPredictor(
        device=device,
        model_path=model_path,
        inference_config=config,
        mix_precision=True  # 启用混合精度（加速推理）
    )
    
    print(f"✓ LimiX 模型初始化完成")
    
    return classifier, device

# ============================================
# 第 7 步：执行推理
# ============================================

def run_inference(classifier, X_train, y_train, X_test):
    """
    使用 LimiX 模型进行推理
    
    推理流程:
    1. 拼接训练集和测试集数据
    2. 预处理（特征编码、去噪等）
    3. 通过 4 个估计器进行集成预测
    4. 返回每个样本的类别概率分布
    
    Args:
        classifier: LimiX 分类器
        X_train, y_train: 训练数据
        X_test: 测试数据
        
    Returns:
        y_pred_proba: 预测概率矩阵 (n_test_samples, n_classes)
        inference_time: 推理耗时（秒）
    """
    print("\n" + "=" * 80)
    print("Step 5: 执行推理")
    print("=" * 80)
    
    print(f"推理中...")
    start_time = time.time()
    
    # 调用 LimiX 预测
    # 注意：LimiX 的 predict 方法需要同时传入训练和测试数据
    y_pred_proba = classifier.predict(
        X_train, y_train, X_test,
        task_type="Classification"
    )
    
    inference_time = time.time() - start_time
    
    print(f"✓ 推理完成")
    print(f"  - 推理时间：{inference_time:.2f} 秒")
    print(f"  - 预测结果形状：{y_pred_proba.shape}")
    
    return y_pred_proba, inference_time

# ============================================
# 第 8 步：评估结果
# ============================================

def evaluate_performance(y_test, y_pred_proba, label_encoder):
    """
    评估模型性能
    
    评估指标:
    - Accuracy: 准确率
    - F1-Score: F1 分数（宏平均和加权平均）
    - AUC-ROC: 曲线下面积
    - Precision: 精确率
    - Recall: 召回率
    
    Args:
        y_test: 真实标签
        y_pred_proba: 预测概率
        label_encoder: 标签编码器
        
    Returns:
        metrics: 各项指标字典
    """
    print("\n" + "=" * 80)
    print("Step 6: 性能评估")
    print("=" * 80)
    
    # 获取预测类别
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # 解码标签（数字 -> 类别名）
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    
    # 计算主要指标
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    # 计算 AUC（多分类）
    # 注意：roc_auc_score 需要概率输入，不支持直接多分类
    # 使用 'ovr' 或 'ovo' 策略
    try:
        auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovo')
    except:
        auc = np.nan
    
    metrics = {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'auc': auc
    }
    
    print(f"\n【整体指标】")
    print(f"  准确率 (Accuracy)   : {acc:.4f}")
    print(f"  F1 分数 (Macro)    : {f1_macro:.4f}")
    print(f"  F1 分数 (Weighted) : {f1_weighted:.4f}")
    if not np.isnan(auc):
        print(f"  AUC-ROC (OvO)     : {auc:.4f}")
    
    # 分类报告
    print(f"\n【分类报告】")
    print(classification_report(y_test_labels, y_pred_labels, digits=4))
    
    # 混淆矩阵
    print(f"\n【混淆矩阵】")
    cm = confusion_matrix(y_test_labels, y_pred_labels)
    print(cm)
    print(f"  类别顺序：{list(label_encoder.classes_)}")
    
    # 置信度分析
    print(f"\n【置信度分析】")
    confidences = np.max(y_pred_proba, axis=1)
    print(f"  平均置信度：{np.mean(confidences):.4f}")
    print(f"  中位数置信度：{np.median(confidences):.4f}")
    print(f"  95% 置信度：{np.percentile(confidences, 95):.4f}")
    
    # 各类别置信度
    print(f"\n【各类别置信度】")
    for i, cls in enumerate(label_encoder.classes_):
        cls_mask = (y_test == i)
        if cls_mask.any():
            cls_conf = confidences[cls_mask]
            print(f"  {cls:12s}: {np.mean(cls_conf):.4f} +/- {np.std(cls_conf):.4f}")
    
    return metrics, y_pred_labels, y_test_labels

# ============================================
# 第 9 步：保存结果
# ============================================

def save_results(y_test_labels, y_pred_labels, y_pred_proba, label_encoder, save_dir='./results'):
    """
    保存测试结果到文件
    
    Args:
        y_test_labels: 真实标签
        y_pred_labels: 预测标签
        y_pred_proba: 预测概率
        label_encoder: 标签编码器
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存预测结果
    confidences = np.max(y_pred_proba, axis=1)
    results_df = pd.DataFrame({
        'true_label': y_test_labels,
        'pred_label': y_pred_labels,
        'confidence': confidences,
        'is_correct': y_test_labels == y_pred_labels
    })
    
    result_path = os.path.join(save_dir, 'detection_results.csv')
    results_df.to_csv(result_path, index=False)
    print(f"\n✓ 结果已保存到：{result_path}")
    
    # 保存统计信息
    stats = {
        'total_samples': int(len(y_test_labels)),
        'correct_samples': int(sum(y_test_labels == y_pred_labels)),
        'accuracy': float(np.mean(y_test_labels == y_pred_labels)),
        'classes': [str(c) for c in label_encoder.classes_]
    }
    
    stats_path = os.path.join(save_dir, 'stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"✓ 统计信息保存到：{stats_path}")

# ============================================
# 主函数
# ============================================

def main():
    """
    主函数：执行完整的故障检测流程
    
    流程:
    1. 加载数据 → 2. 特征工程 → 3. 数据预处理
    4. 初始化模型 → 5. 执行推理 → 6. 评估结果
    7. 保存结果
    """
    print("\n" + "=" * 80)
    print("LimiX 温振传感器故障检测演示")
    print("=" * 80)
    
    # 检查文件是否存在
    if not os.path.exists(DATA_PATH):
        print(f"\n❌ 错误：找不到数据文件：{DATA_PATH}")
        print("请先运行数据生成脚本:")
        print("  python limix_fault_detection/data/generate_sensor_data.py")
        return
    
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ 错误：找不到模型文件：{MODEL_PATH}")
        return
    
    # Step 1: 加载数据
    df = load_data(DATA_PATH)
    
    # Step 2: 特征工程
    X, y, feature_names, label_encoder = prepare_features(df, WINDOW_SIZE, SLIDE_STEP)
    
    # Step 3: 数据预处理
    X_train, X_test, y_train, y_test, scaler = preprocess_data(X, y)
    
    # Step 4: 初始化模型
    classifier, device = init_model(MODEL_PATH, CONFIG_PATH)
    
    # Step 5: 执行推理
    y_pred_proba, inference_time = run_inference(classifier, X_train, y_train, X_test)
    
    # Step 6: 评估结果
    metrics, y_pred_labels, y_test_labels = evaluate_performance(y_test, y_pred_proba, label_encoder)
    
    # Step 7: 保存结果
    save_results(y_test_labels, y_pred_labels, y_pred_proba, label_encoder)
    
    # 总结
    print("\n" + "=" * 80)
    print("演示完成!")
    print("=" * 80)
    print(f"\n【系统配置】")
    print(f"  • 模型版本：LimiX-2M")
    print(f"  • 设备：{device}")
    print(f"  • 特征维度：{len(feature_names)}")
    print(f"  • 故障类型：{len(label_encoder.classes_)} 类")
    print(f"\n【性能指标】")
    print(f"  • 推理耗时：{inference_time:.2f} 秒")
    print(f"  • 准确率：{metrics['accuracy']:.4f}")
    print(f"  • F1 分数：{metrics['f1_macro']:.4f}")
    print(f"\n【文件输出】")
    print(f"  • 预测结果：./results/detection_results.csv")
    print(f"  • 统计信息：./results/stats.json")
    
    return metrics

# ============================================
# 运行
# ============================================

if __name__ == '__main__':
    metrics = main()
    print("\n✅ 脚本执行完成!")
