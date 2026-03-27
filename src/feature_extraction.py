import numpy as np
import pandas as pd
from typing import Dict, List

class LowFreqFeatureExtractor:
    """低频采样数据特征提取器"""

    def __init__(self, window_size: int = 60, slide_step: int = 1):
        self.window_size = window_size
        self.slide_step = slide_step

    def extract_features(self, data_window: pd.DataFrame) -> Dict[str, float]:
        """
        提取滑动窗口特征
        
        Args:
            data_window: DataFrame with shape (window_size, n_features)
            
        Returns:
            dict of features
        """
        features = {}

        # 1. 基础统计特征
        for col in data_window.columns:
            if 'temp' in col or 'vib' in col:
                values = data_window[col].values
                features[f'{col}_mean'] = float(np.mean(values))
                features[f'{col}_std'] = float(np.std(values))
                features[f'{col}_min'] = float(np.min(values))
                features[f'{col}_max'] = float(np.max(values))
                features[f'{col}_range'] = float(np.max(values) - np.min(values))
                features[f'{col}_median'] = float(np.median(values))

                # 2. 趋势特征
                if len(values) > 1:
                    trend = np.polyfit(np.arange(len(values)), values, 1)[0]
                    features[f'{col}_trend'] = float(trend)

                # 3. 变化率特征
                if len(values) > 1:
                    features[f'{col}_change_rate'] = float((values[-1] - values[0]) / (abs(values[0]) + 1e-6))
                    features[f'{col}_diff_mean'] = float(np.mean(np.diff(values)))

                # 4. 高阶统计特征
                if len(values) > 3:
                    features[f'{col}_skew'] = float(np.mean(((values - np.mean(values)) / (np.std(values) + 1e-6)) ** 3))
                    features[f'{col}_kurtosis'] = float(np.mean(((values - np.mean(values)) / (np.std(values) + 1e-6)) ** 4) - 3)

                # 5. 异常特征
                mean_val = np.mean(values)
                std_val = np.std(values) + 1e-6
                z_scores = np.abs((values - mean_val) / std_val)
                features[f'{col}_anomaly_count'] = float(np.sum(z_scores > 2.0))
                features[f'{col}_anomaly_ratio'] = float(np.sum(z_scores > 2.0) / len(values))
                features[f'{col}_max_zscore'] = float(np.max(z_scores))

        # 6. 传感器融合特征
        temp_cols = [c for c in data_window.columns if 'temp' in c]
        vib_cols = [c for c in data_window.columns if 'vib' in c]

        if temp_cols and vib_cols:
            temp_values = np.mean(data_window[temp_cols].values, axis=1)
            vib_values = np.mean(data_window[vib_cols].values, axis=1)
            
            # 温度 - 振动相关性
            features['temp_vib_correlation'] = float(np.corrcoef(temp_values, vib_values)[0, 1]) if len(temp_values) > 1 else 0.0
            features['temp_mean_overall'] = float(np.mean(temp_values))
            features['vib_mean_overall'] = float(np.mean(vib_values))
            features['vib_temp_ratio'] = float(np.mean(vib_values) / (np.mean(temp_values) + 1e-6))
            
            # 温度变化与振动变化的关系
            if len(temp_values) > 1:
                temp_diff = np.diff(temp_values)
                vib_diff = np.diff(vib_values)
                features['temp_vib_diff_corr'] = float(np.corrcoef(temp_diff, vib_diff)[0, 1]) if len(temp_diff) > 1 else 0.0

        # 7. 振动方向特征
        if len(vib_cols) >= 3:
            vib_x_mean = np.mean(data_window[vib_cols[0]].values)
            vib_y_mean = np.mean(data_window[vib_cols[1]].values)
            vib_z_mean = np.mean(data_window[vib_cols[2]].values)
            
            features['vib_total'] = float(np.sqrt(vib_x_mean**2 + vib_y_mean**2 + vib_z_mean**2))
            features['vib_x_dominance'] = float(vib_x_mean / (vib_y_mean + 1e-6))
            features['vib_z_dominance'] = float(vib_z_mean / (vib_y_mean + 1e-6))

        # 8. 温度差异特征
        if len(temp_cols) >= 2:
            temp1_mean = np.mean(data_window[temp_cols[0]].values)
            temp2_mean = np.mean(data_window[temp_cols[1]].values)
            features['temp_diff'] = float(abs(temp1_mean - temp2_mean))
            features['temp_ratio'] = float(temp1_mean / (temp2_mean + 1e-6))

        return features

def prepare_windowed_dataset(df: pd.DataFrame, 
                           window_size: int = 30,
                           slide_step: int = 10) -> tuple:
    """
    从时间序列数据准备带标签的窗口数据集
    
    Args:
        df: 原始数据 DataFrame
        window_size: 窗口大小
        slide_step: 滑动步长
        
    Returns:
        X: 特征 numpy 数组
        y: 标签 numpy 数组  
        feature_names: 特征名列表
        label_encoder: 标签编码器
    """
    from sklearn.preprocessing import LabelEncoder
    
    extractor = LowFreqFeatureExtractor(window_size=window_size)
    
    feature_list = []
    label_list = []
    timestamp_list = []
    
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'device_id', 'fault_label']]
    
    # 按窗口提取特征
    for i in range(window_size, len(df), slide_step):
        window = df.iloc[i-window_size:i][feature_cols]
        label = df.iloc[i]['fault_label']
        timestamp = df.iloc[i]['timestamp']
        
        features = extractor.extract_features(window)
        feature_list.append(features)
        label_list.append(label)
        timestamp_list.append(timestamp)
    
    # 转换为 DataFrame
    X_df = pd.DataFrame(feature_list)
    y = np.array(label_list)
    
    # 编码标签
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    feature_names = list(X_df.columns)
    X = X_df.values.astype(np.float32)
    
    print(f"数据集准备完成:")
    print(f"  样本数：{len(X)}")
    print(f"  特征数：{X.shape[1]}")
    print(f"  标签类别：{label_encoder.classes_}")
    print(f"  各类别样本数:")
    for cls in label_encoder.classes_:
        count = np.sum(y == label_encoder.transform([cls])[0])
        print(f"    {cls}: {count}")
    
    return X, y_encoded, feature_names, label_encoder

if __name__ == '__main__':
    # 测试特征提取
    df = pd.read_csv('limix_fault_detection/data/raw/sensor_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    X, y, features, le = prepare_windowed_dataset(df, window_size=60, slide_step=60)
    print(f"\n前 5 个特征：{features[:5]}")
    print(f"\n标签映射：{dict(zip(le.classes_, le.transform(le.classes_)))}")