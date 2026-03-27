#!/usr/bin/env python
"""
温振传感器故障检测系统测试脚本
基于 LimiX 模型的多分类故障检测
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    f1_score
)
from sklearn.preprocessing import MinMaxScaler

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.predictor import LimiXPredictor
from src.feature_extraction import (
    LowFreqFeatureExtractor,
    prepare_windowed_dataset
)


def load_and_prepare_data(csv_path, window_size=60, slide_step=60):
    """
    加载数据并准备窗口特征
    
    Args:
        csv_path: 原始 CSV 数据路径
        window_size: 滑动窗口大小
        slide_step: 滑动步长
        
    Returns:
        X_train, X_test, y_train, y_test, feature_names, label_encoder, scaler
    """
    print("=" * 60)
    print("Step 1: 加载原始数据")
    print("=" * 60)
    
    # 加载 CSV 数据
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"原始数据形状：{df.shape}")
    print(f"时间范围：{df['timestamp'].min()} 到 {df['timestamp'].max()}")
    print(f"\n数据列：{list(df.columns)}")
    print(f"\n标签分布:")
    print(df['fault_label'].value_counts())
    
    # 提取窗口特征
    print("\n" + "=" * 60)
    print("Step 2: 提取滑动窗口特征")
    print("=" * 60)
    
    X, y, feature_names, label_encoder = prepare_windowed_dataset(
        df, 
        window_size=window_size, 
        slide_step=slide_step
    )
    
    # 处理可能的 NaN 值
    nan_mask = np.isnan(X)
    if nan_mask.any():
        nan_count = nan_mask.sum()
        print(f"\n警告：发现 {nan_count} 个 NaN 值，进行中值填充...")
        for col in range(X.shape[1]):
            col_nans = nan_mask[:, col]
            if col_nans.any():
                median_val = np.nanmedian(X[:, col])
                X[col_nans, col] = median_val
    
    # 归一化
    print("\n归一化数据...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    
    # 划分训练集/测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print(f"\n训练集大小：{X_train.shape}")
    print(f"测试集大小：{X_test.shape}")
    print(f"\n各类别分布:")
    for cls in label_encoder.classes_:
        train_count = np.sum(y_train == label_encoder.transform([cls])[0])
        test_count = np.sum(y_test == label_encoder.transform([cls])[0])
        print(f"  {cls:12s} - 训练：{train_count:3d}, 测试：{test_count:3d}")
    
    return X_train, X_test, y_train, y_test, feature_names, label_encoder, scaler, df


def train_and_evaluate(X_train, y_train, X_test, y_test, 
                       model_path, config_path, label_encoder):
    """
    使用 LimiX 模型进行推理和评估
    
    Args:
        X_train, y_train: 训练数据
        X_test, y_test: 测试数据
        model_path: LimiX 模型路径
        config_path: 推理配置路径
        label_encoder: 标签编码器
        
    Returns:
        classifier: 训练好的分类器
    """
    print("\n" + "=" * 60)
    print("Step 3: 初始化 LimiX 模型")
    print("=" * 60)
    
    # 检查 GPU 可用性
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备：{device}")
    print(f"CUDA 版本：{torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    print(f"PyTorch 版本：{torch.__version__}")
    
    # 加载配置
    print(f"\n配置文件：{config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"估计器数量：{len(config)}")
    
    # 初始化分类器
    print(f"\n加载模型：{model_path}")
    classifier = LimiXPredictor(
        device=device,
        model_path=model_path,
        inference_config=config,
        mix_precision=True
    )
    
    # 进行推理
    print("\n" + "=" * 60)
    print("Step 4: 模型推理")
    print("=" * 60)
    
    start_time = time.time()
    
    print("开始推理...")
    y_pred_proba = classifier.predict(
        X_train, y_train, X_test,
        task_type="Classification"
    )
    
    inference_time = time.time() - start_time
    print(f"推理完成! 耗时：{inference_time:.2f} 秒")
    
    # 获取预测结果
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # 解码标签
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    
    # 评估
    print("\n" + "=" * 60)
    print("Step 5: 评估结果")
    print("=" * 60)
    
    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n整体指标:")
    print(f"  准确率 (Accuracy): {acc:.4f}")
    print(f"  F1 分数 (Macro):   {f1_macro:.4f}")
    print(f"  F1 分数 (Weighted): {f1_weighted:.4f}")
    
    print(f"\n分类报告:")
    print(classification_report(y_test_labels, y_pred_labels, digits=4))
    
    print(f"\n混淆矩阵:")
    cm = confusion_matrix(y_test_labels, y_pred_labels)
    print(cm)
    
    # 详细分析
    print(f"\n各类别详细分析:")
    for i, cls in enumerate(label_encoder.classes_):
        cls_mask = (y_test == i)
        pred_mask = (y_pred == i)
        
        tp = np.sum(cls_mask & pred_mask)
        fn = np.sum(cls_mask & ~pred_mask)
        fp = np.sum(~cls_mask & pred_mask)
        tn = np.sum(~cls_mask & ~pred_mask)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n{cls:15s}:")
        print(f"  TP: {tp:3d}, FN: {fn:3d}, FP: {fp:3d}")
        print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        # 置信度分析
        if cls_mask.any():
            pred_conf = np.max(y_pred_proba[cls_mask], axis=1)
            print(f"  平均置信度：{np.mean(pred_conf):.4f} (min: {np.min(pred_conf):.4f}, max: {np.max(pred_conf):.4f})")
            print(f"  中位数置信度：{np.median(pred_conf):.4f}")
    
    # 置信度分布
    print(f"\n所有预测置信度分布:")
    all_conf = np.max(y_pred_proba, axis=1)
    print(f"  平均：{np.mean(all_conf):.4f}")
    print(f"  中位数：{np.median(all_conf):.4f}")
    print(f"  标准差：{np.std(all_conf):.4f}")
    print(f"  90% 置信度阈值：{np.percentile(all_conf, 90):.4f}")
    print(f"  95% 置信度阈值：{np.percentile(all_conf, 95):.4f}")
    
    return classifier


def save_results(X_train, y_train, y_test, y_pred_proba, label_encoder, 
                save_dir='./limix_fault_detection/results'):
    """
    保存评估结果
    
    Args:
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存预测结果
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    confidences = np.max(y_pred_proba, axis=1)
    
    results_df = pd.DataFrame({
        'true_label': y_test_labels,
        'pred_label': y_pred_labels,
        'confidence': confidences
    })
    
    result_path = os.path.join(save_dir, 'detection_results.csv')
    results_df.to_csv(result_path, index=False)
    print(f"\n结果已保存到：{result_path}")
    
    # 保存置信度分布
    for cls in label_encoder.classes_:
        cls_idx = label_encoder.transform([cls])[0]
        cls_conf = confidences[y_test == cls_idx]
        stats = {
            'class': cls,
            'count': len(cls_conf),
            'mean_confidence': np.mean(cls_conf),
            'median_confidence': np.median(cls_conf),
            'std_confidence': np.std(cls_conf),
            'min_confidence': np.min(cls_conf),
            'max_confidence': np.max(cls_conf)
        }
        print(f"  {cls}: {stats['mean_confidence']:.4f} +/- {stats['std_confidence']:.4f}")


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("LimiX 温振传感器故障检测系统")
    print("=" * 60)
    
    # 路径配置
    DATA_PATH = './limix_fault_detection/data/raw/sensor_data.csv'
    MODEL_PATH = './LimiX-2M.ckpt'  # 使用 2M 模型进行快速推理
    CONFIG_PATH = './config/cls_default_2M_retrieval.json'
    
    # 检查文件存在
    if not os.path.exists(DATA_PATH):
        print(f"错误：找不到数据文件：{DATA_PATH}")
        print("请先运行数据生成脚本:")
        print("  python limix_fault_detection/data/generate_sensor_data.py")
        return
    
    if not os.path.exists(MODEL_PATH):
        print(f"错误：找不到模型文件：{MODEL_PATH}")
        return
    
    if not os.path.exists(CONFIG_PATH):
        print(f"错误：找不到配置文件：{CONFIG_PATH}")
        return
    
    # 窗口参数
    WINDOW_SIZE = 30   # 窗口大小
    SLIDE_STEP = 10    # 滑动步长（小步长 = 更多样本）
    
    # 数据准备
    X_train, X_test, y_train, y_test, feature_names, label_encoder, scaler, raw_df = (
        load_and_prepare_data(DATA_PATH, window_size=WINDOW_SIZE, slide_step=SLIDE_STEP)
    )
    
    # 训练和评估
    classifier = train_and_evaluate(
        X_train, y_train, X_test, y_test,
        MODEL_PATH, CONFIG_PATH, label_encoder
    )
    
    # 保存结果
    y_pred_proba = classifier.predict(X_train, y_train, X_test, task_type="Classification")
    save_results(X_train, y_train, y_test, y_pred_proba, label_encoder)
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()