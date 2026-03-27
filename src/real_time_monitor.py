import sys
import time
import numpy as np
import pandas as pd
from collections import deque
from datetime import datetime
import torch
from typing import Dict, List, Optional, Tuple
import json
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.predictor import LimiXPredictor
from src.feature_extraction import LowFreqFeatureExtractor


class SmartAlarmStrategy:
    """智能报警策略"""

    def __init__(self,
                 min_confidence: float = 0.8,
                 alarm_duration_threshold: int = 3,
                 alarm_cooldown: int = 60):
        """
        Args:
            min_confidence: 最小置信度阈值
            alarm_duration_threshold: 连续报警次数阈值
            alarm_cooldown: 报警冷却期（秒）
        """
        self.min_confidence = min_confidence
        self.alarm_duration_threshold = alarm_duration_threshold
        self.alarm_cooldown = alarm_cooldown
        
        self.alarm_counter = {}
        self.last_alarm_time = {}
        self.alarm_history = []

    def check_alarm(self, pred_class: str, confidence: float, device_id: str = 'device_001') -> Optional[Dict]:
        """
        智能报警检查
        
        Args:
            pred_class: 预测的故障类型
            confidence: 置信度
            device_id: 设备 ID
            
        Returns:
            报警信息字典，如果无需报警返回 None
        """
        now = time.time()
        
        # 检查置信度
        if confidence < self.min_confidence:
            self.alarm_counter[device_id] = 0
            return None
        
        # 正常状态，重置计数器
        if pred_class == "正常":
            self.alarm_counter[device_id] = 0
            return None
        
        # 检查冷却期
        if (device_id in self.last_alarm_time and 
            now - self.last_alarm_time[device_id] < self.alarm_cooldown):
            return None
        
        # 故障状态，计数
        if device_id not in self.alarm_counter:
            self.alarm_counter[device_id] = 0
        
        self.alarm_counter[device_id] += 1
        
        # 达到阈值，触发报警
        if self.alarm_counter[device_id] >= self.alarm_duration_threshold:
            self.last_alarm_time[device_id] = now
            self.alarm_counter[device_id] = 0
            
            alarm = {
                'timestamp': datetime.now(),
                'device_id': device_id,
                'fault_type': pred_class,
                'confidence': confidence,
                'duration': self.alarm_counter[device_id]
            }
            
            self.alarm_history.append(alarm)
            
            return alarm
        
        return None


class RealTimeFaultMonitor:
    """实时故障监控系统"""

    def __init__(self,
                 model_path: str,
                 config_path: str,
                 X_train: np.ndarray,
                 y_train: np.ndarray,
                 label_encoder,
                 scaler,
                 window_size: int = 60,
                 confidence_threshold: float = 0.8,
                 device: torch.device = None):
        """
        Args:
            model_path: LimiX 模型路径
            config_path: 推理配置路径
            X_train: 训练特征数据
            y_train: 训练标签数据
            label_encoder: 标签编码器
            scaler: 数据归一化器
            window_size: 滑动窗口大小
            confidence_threshold: 置信度阈值
            device: 推理设备
        """
        # 加载配置
        with open(config_path, 'r') as f:
            inference_config = json.load(f)
        
        # 初始化设备
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化模型
        self.model = LimiXPredictor(
            device=device,
            model_path=model_path,
            inference_config=inference_config,
            mix_precision=True
        )
        
        self.label_encoder = label_encoder
        self.scaler = scaler
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold
        
        # 数据缓冲区
        self.buffer = deque(maxlen=window_size)
        
        # 报警策略
        self.alarm_strategy = SmartAlarmStrategy(
            min_confidence=confidence_threshold,
            alarm_duration_threshold=3,
            alarm_cooldown=60
        )
        
        # 特征提取器
        self.feature_extractor = LowFreqFeatureExtractor(window_size=window_size)
        
        # 保存训练数据（LimiX 需要训练数据用于检索）
        self.X_train = X_train
        self.y_train = y_train
        
        # 特征列名
        self.feature_cols = ['temp_1', 'temp_2', 'vib_x', 'vib_y', 'vib_z']
        
        # 统计信息
        self.prediction_count = 0
        self.alarm_count = 0
        
        print(f"实时监控器初始化完成")
        print(f"  设备：{device}")
        print(f"  窗口大小：{window_size}")
        print(f"  置信度阈值：{confidence_threshold}")
        print(f"  训练样本数：{len(X_train)}")

    def add_data_point(self, data_point: Dict) -> Tuple[Optional[str], float, Optional[Dict]]:
        """
        添加新的数据点并进行预测
        
        Args:
            data_point: 数据点字典，包含 sensor 数据
            
        Returns:
            (预测类型，置信度，报警信息)
        """
        # 添加到缓冲区
        self.buffer.append(data_point)
        
        # 缓冲区未满，不进行预测
        if len(self.buffer) < self.window_size:
            return None, 0.0, None
        
        # 提取特征
        window_df = pd.DataFrame(list(self.buffer))
        features = self.feature_extractor.extract_features(window_df)
        
        # 转换为数组并归一化
        feature_names = sorted(features.keys())
        features_array = np.array([list(features[f] for f in feature_names)])
        features_normalized = self.scaler.transform(features_array)
        
        # 预测
        y_pred_proba = self.model.predict(
            self.X_train, self.y_train,
            features_normalized,
            task_type="Classification"
        )
        
        # 解析结果
        pred_class_idx = np.argmax(y_pred_proba, axis=1)[0]
        confidence = np.max(y_pred_proba, axis=1)[0]
        pred_class = self.label_encoder.inverse_transform([pred_class_idx])[0]
        
        # 统计
        self.prediction_count += 1
        
        # 置信度过低
        if confidence < self.confidence_threshold:
            return pred_class, confidence, None
        
        # 报警逻辑
        device_id = data_point.get('device_id', 'device_001')
        alarm = self.alarm_strategy.check_alarm(pred_class, confidence, device_id)
        
        if alarm:
            self.alarm_count += 1
        
        return pred_class, confidence, alarm

    def simulate_real_time_stream(self, data_generator, duration: int = None):
        """
        模拟实时数据流
        
        Args:
            data_generator: 数据生成器函数
            duration: 运行时长（秒）
        """
        import time
        
        start_time = time.time()
        print("\n开始实时监控...")
        print("=" * 80)
        
        while True:
            # 检查是否超时
            if duration and (time.time() - start_time) > duration:
                break
            
            # 获取数据点
            data_point = next(data_generator)
            
            # 添加数据点
            pred_class, confidence, alarm = self.add_data_point(data_point)
            
            # 显示结果
            if pred_class is not None:
                status = "正常" if pred_class == "正常" else f"⚠️ {pred_class}"
                confidence_str = f"{confidence:.4f}"
                
                if alarm:
                    print(f"[ALARM] 时间：{alarm['timestamp'].strftime('%H:%M:%')} | "
                          f"故障：{alarm['fault_type']} | 置信度：{alarm['confidence']:.4f}")
                else:
                    print(f"监控：{status} (置信度：{confidence:.4f})")
            
            # 控制频率
            time.sleep(0.1)  # 模拟数据间隔

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        return {
            'total_predictions': self.prediction_count,
            'total_alarms': self.alarm_count,
            'alarm_rate': self.alarm_count / self.prediction_count if self.prediction_count > 0 else 0,
            'buffer_size': len(self.buffer),
            'alarm_history_len': len(self.alarm_strategy.alarm_history)
        }


def simulate_sensor_data_stream():
    """模拟传感器数据流生成器"""
    import numpy as np
    
    base_temp1 = 45.0
    base_temp2 = 46.0
    base_vib = 0.13
    
    fault_mode = 'normal'
    fault_timer = 0
    
    while True:
        # 生成数据点
        if fault_mode == 'normal':
            temp1 = base_temp1 + np.random.normal(0, 0.3)
            temp2 = base_temp2 + np.random.normal(0, 0.3)
            vib_x = base_vib + np.random.normal(0, 0.01)
            vib_y = base_vib + np.random.normal(0, 0.01)
            vib_z = base_vib + np.random.normal(0, 0.01)
        elif fault_mode == 'bearing':
            temp1 = base_temp1 + 3 + np.random.normal(0, 0.4)
            temp2 = base_temp2 + 3 + np.random.normal(0, 0.4)
            vib_x = base_vib + 0.12 + np.random.normal(0, 0.03)
            vib_y = base_vib + 0.12 + np.random.normal(0, 0.03)
            vib_z = base_vib + 0.12 + np.random.normal(0, 0.03)
        elif fault_mode == 'gear':
            temp1 = base_temp1 + 5 + np.random.normal(0, 0.5)
            temp2 = base_temp2 + 5 + np.random.normal(0, 0.5)
            vib_x = base_vib + 0.17 + np.random.normal(0, 0.04)
            vib_y = base_vib + 0.17 + np.random.normal(0, 0.04)
            vib_z = base_vib + 0.17 + np.random.normal(0, 0.04)
        
        fault_timer += 1
        if fault_timer > 100 and fault_mode == 'normal':
            fault_mode = np.random.choice(['bearing', 'gear'])
            fault_timer = 0
        elif fault_timer > 100 and fault_mode != 'normal':
            fault_mode = 'normal'
            fault_timer = 0
        
        yield {
            'device_id': 'device_001',
            'temp_1': round(temp1, 2),
            'temp_2': round(temp2, 2),
            'vib_x': round(max(0, vib_x), 4),
            'vib_y': round(max(0, vib_y), 4),
            'vib_z': round(max(0, vib_z), 4)
        }


if __name__ == '__main__':
    # 测试
    print("实时监控模块加载成功")
