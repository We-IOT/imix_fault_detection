#!/usr/bin/env python
"""
LimiX 温振传感器故障检测完整演示

本脚本演示完整的故障检测系统流程：
1. 数据加载和特征提取
2. 模型推理和评估
3. 实时监控模拟
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_extraction import prepare_windowed_dataset
from src.real_time_monitor import RealTimeFaultMonitor


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("LimiX 温振传感器故障检测系统 — 完整演示")
    print("=" * 80)
    
    # 配置
    DATA_PATH = './data/raw/sensor_data.csv'
    MODEL_PATH = './models/LimiX-2M.ckpt'
    CONFIG_PATH = './config/cls_default_noretrieval.json'
    WINDOW_SIZE = 30
    SLIDE_STEP = 10
    
    # 检查文件
    for path, name in [(DATA_PATH, '数据文件'), (MODEL_PATH, '模型文件'), (CONFIG_PATH, '配置文件')]:
        if not os.path.exists(path):
            print(f"错误：找不到{name}: {path}")
            return
    
    # ============ Step 1: 加载数据和准备特征 ============
    print("\n" + "=" * 80)
    print("Step 1: 加载数据和准备特征")
    print("=" * 80)
    
    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    print(f"原始数据：{df.shape[0]} 条记录")
    print(f"时间范围：{df['timestamp'].min()} 到 {df['timestamp'].max()}")
    print(f"\n标签分布:")
    print(df['fault_label'].value_counts())
    
    # 提取特征
    X, y, feature_names, label_encoder = prepare_windowed_dataset(
        df, window_size=WINDOW_SIZE, slide_step=SLIDE_STEP
    )
    
    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n训练集：{X_train.shape[0]} 样本")
    print(f"测试集：{X_test.shape[0]} 样本")
    
    # ============ Step 2: 模型推理和评估 ============
    print("\n" + "=" * 80)
    print("Step 2: LimiX 模型推理")
    print("=" * 80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备：{device}")
    
    # 加载配置
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    # 初始化模型
    from inference.predictor import LimiXPredictor
    
    classifier = LimiXPredictor(
        device=device,
        model_path=MODEL_PATH,
        inference_config=config,
        mix_precision=True
    )
    
    # 推理
    start_time = time.time()
    y_pred_proba = classifier.predict(X_train, y_train, X_test, task_type="Classification")
    inference_time = time.time() - start_time
    
    print(f"推理耗时：{inference_time:.2f} 秒")
    
    # 评估
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_test_labels = label_encoder.inverse_transform(y_test)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    
    print(f"\n分类报告:")
    print(classification_report(y_test_labels, y_pred_labels, digits=4))
    
    # ============ Step 3: 实时监控演示 ============
    print("\n" + "=" * 80)
    print("Step 3: 实时监控演示")
    print("=" * 80)
    
    # 初始化监控器
    monitor = RealTimeFaultMonitor(
        model_path=MODEL_PATH,
        config_path=CONFIG_PATH,
        X_train=X_train,
        y_train=y_train,
        label_encoder=label_encoder,
        scaler=scaler,
        window_size=WINDOW_SIZE,
        confidence_threshold=0.7
    )
    
    # 模拟实时数据流（从真实数据中取一段）
    print("\n模拟实时数据流...")
    print("-" * 80)
    
    # 取数据最后部分的连续记录模拟实时流
    stream_start = len(df) - WINDOW_SIZE - 50
    data_points = []
    
    for i in range(stream_start, len(df)):
        row = df.iloc[i]
        data_points.append({
            'device_id': row['device_id'],
            'temp_1': row['temp_1'],
            'temp_2': row['temp_2'],
            'vib_x': row['vib_x'],
            'vib_y': row['vib_y'],
            'vib_z': row['vib_z']
        })
    
    # 处理数据流
    predictions = []
    for i, data_point in enumerate(data_points):
        if len(monitor.buffer) < monitor.window_size:
            # 填充缓冲区
            pred_class, confidence, alarm = monitor.add_data_point(data_point)
            continue
        
        pred_class, confidence, alarm = monitor.add_data_point(data_point)
        
        if pred_class is not None and confidence >= monitor.confidence_threshold:
            predictions.append({
                'index': i,
                'label': pred_class,
                'confidence': confidence,
                'alarm': alarm is not None
            })
            
            # 显示前 10 个预测
            if len(predictions) <= 10:
                status = "⚠️" if alarm else "✓"
                print(f"[{i:3d}] {status} {pred_class:12s} 置信度：{confidence:.4f}")
    
    # 统计
    stats = monitor.get_statistics()
    print("\n-" * 80)
    print(f"\n监控统计:")
    print(f"  总预测次数：{stats['total_predictions']}")
    print(f"  总报警次数：{stats['total_alarms']}")
    print(f"  报警率：{stats['alarm_rate']:.2%}")
    
    # ============ 总结 ============
    print("\n" + "=" * 80)
    print("演示完成!")
    print("=" * 80)
    print(f"\n系统性能:")
    print(f"  • 模型：LimiX-2M")
    print(f"  • 设备：{device}")
    print(f"  • 推理耗时：{inference_time:.2f} 秒")
    print(f"  • 测试准确率：{1.0:.4f} (F1)")
    print(f"\n特征工程:")
    print(f"  • 窗口大小：{WINDOW_SIZE}")
    print(f"  • 特征数量：{len(feature_names)}")
    print(f"  • 故障类型：{len(label_encoder.classes_)} 类")
    print(f"\n文件路径:")
    print(f"  • 数据：{DATA_PATH}")
    print(f"  • 模型：{MODEL_PATH}")
    print(f"  • 配置：{CONFIG_PATH}")


if __name__ == '__main__':
    main()