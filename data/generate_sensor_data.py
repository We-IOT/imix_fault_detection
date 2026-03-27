import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

def generate_normal_data(n_samples, start_time):
    """生成正常状态数据"""
    data = []
    base_temp1 = 45.0
    base_temp2 = 46.0
    base_vib = 0.13
    
    for i in range(n_samples):
        timestamp = start_time + timedelta(minutes=i)
        
        # 正常状态：小幅度波动
        temp1 = base_temp1 + np.random.normal(0, 0.3)
        temp2 = base_temp2 + np.random.normal(0, 0.3)
        vib_x = base_vib + np.random.normal(0, 0.01)
        vib_y = base_vib + np.random.normal(0, 0.01)
        vib_z = base_vib + np.random.normal(0, 0.01)
        
        data.append({
            'timestamp': timestamp,
            'device_id': 'device_001',
            'temp_1': round(temp1, 2),
            'temp_2': round(temp2, 2),
            'vib_x': round(max(0, vib_x), 4),
            'vib_y': round(max(0, vib_y), 4),
            'vib_z': round(max(0, vib_z), 4),
            'fault_label': '正常'
        })
    
    return data

def generate_bearing_fault_data(n_samples, start_time):
    """生成轴承故障数据"""
    data = []
    base_temp1 = 48.0
    base_temp2 = 49.0
    base_vib = 0.25
    
    for i in range(n_samples):
        timestamp = start_time + timedelta(minutes=i)
        
        # 轴承故障：振动明显增加
        temp1 = base_temp1 + np.random.normal(0, 0.4) + i * 0.01
        temp2 = base_temp2 + np.random.normal(0, 0.4) + i * 0.01
        vib_x = base_vib + np.random.normal(0, 0.03)
        vib_y = base_vib + np.random.normal(0, 0.03)
        vib_z = base_vib + np.random.normal(0, 0.03)
        
        data.append({
            'timestamp': timestamp,
            'device_id': 'device_001',
            'temp_1': round(temp1, 2),
            'temp_2': round(temp2, 2),
            'vib_x': round(max(0, vib_x), 4),
            'vib_y': round(max(0, vib_y), 4),
            'vib_z': round(max(0, vib_z), 4),
            'fault_label': '轴承故障'
        })
    
    return data

def generate_gear_fault_data(n_samples, start_time):
    """生成齿轮故障数据"""
    data = []
    base_temp1 = 50.0
    base_temp2 = 51.0
    base_vib = 0.30
    
    for i in range(n_samples):
        timestamp = start_time + timedelta(minutes=i)
        
        # 齿轮故障：振动更高，有周期性波动
        temp1 = base_temp1 + np.random.normal(0, 0.5) + i * 0.015
        temp2 = base_temp2 + np.random.normal(0, 0.5) + i * 0.015
        vib_x = base_vib + np.sin(i * 0.1) * 0.05 + np.random.normal(0, 0.04)
        vib_y = base_vib + np.sin(i * 0.1 + 2) * 0.05 + np.random.normal(0, 0.04)
        vib_z = base_vib + np.sin(i * 0.1 + 4) * 0.05 + np.random.normal(0, 0.04)
        
        data.append({
            'timestamp': timestamp,
            'device_id': 'device_001',
            'temp_1': round(temp1, 2),
            'temp_2': round(temp2, 2),
            'vib_x': round(max(0, vib_x), 4),
            'vib_y': round(max(0, vib_y), 4),
            'vib_z': round(max(0, vib_z), 4),
            'fault_label': '齿轮故障'
        })
    
    return data

def generate_overheat_data(n_samples, start_time):
    """生成过热故障数据"""
    data = []
    base_temp1 = 55.0
    base_temp2 = 56.0
    base_vib = 0.15
    
    for i in range(n_samples):
        timestamp = start_time + timedelta(minutes=i)
        
        # 过热：温度明显升高，振动轻微增加
        temp1 = base_temp1 + np.random.normal(0, 0.6) + i * 0.02
        temp2 = base_temp2 + np.random.normal(0, 0.6) + i * 0.02
        vib_x = base_vib + np.random.normal(0, 0.015)
        vib_y = base_vib + np.random.normal(0, 0.015)
        vib_z = base_vib + np.random.normal(0, 0.015)
        
        data.append({
            'timestamp': timestamp,
            'device_id': 'device_001',
            'temp_1': round(temp1, 2),
            'temp_2': round(temp2, 2),
            'vib_x': round(max(0, vib_x), 4),
            'vib_y': round(max(0, vib_y), 4),
            'vib_z': round(max(0, vib_z), 4),
            'fault_label': '过热'
        })
    
    return data

def generate_imbalance_data(n_samples, start_time):
    """生成不平衡故障数据"""
    data = []
    base_temp1 = 47.0
    base_temp2 = 48.0
    base_vib = 0.20
    
    for i in range(n_samples):
        timestamp = start_time + timedelta(minutes=i)
        
        # 不平衡：特定方向振动增加
        temp1 = base_temp1 + np.random.normal(0, 0.35)
        temp2 = base_temp2 + np.random.normal(0, 0.35)
        vib_x = base_vib + np.random.normal(0, 0.025)  # X 方向振动明显
        vib_y = base_vib * 0.8 + np.random.normal(0, 0.02)
        vib_z = base_vib * 0.8 + np.random.normal(0, 0.02)
        
        data.append({
            'timestamp': timestamp,
            'device_id': 'device_001',
            'temp_1': round(temp1, 2),
            'temp_2': round(temp2, 2),
            'vib_x': round(max(0, vib_x), 4),
            'vib_y': round(max(0, vib_y), 4),
            'vib_z': round(max(0, vib_z), 4),
            'fault_label': '不平衡'
        })
    
    return data

def generate_dataset():
    """生成完整数据集"""
    print("开始生成模拟温振传感器数据...")
    
    all_data = []
    current_time = datetime(2024, 1, 1, 0, 0, 0)
    
    # 正常数据 (400 条)
    print("  - 生成正常数据...")
    all_data.extend(generate_normal_data(400, current_time))
    current_time += timedelta(minutes=400)
    
    # 轴承故障 (200 条)
    print("  - 生成轴承故障数据...")
    all_data.extend(generate_bearing_fault_data(200, current_time))
    current_time += timedelta(minutes=200)
    
    # 齿轮故障 (200 条)
    print("  - 生成齿轮故障数据...")
    all_data.extend(generate_gear_fault_data(200, current_time))
    current_time += timedelta(minutes=200)
    
    # 过热故障 (150 条)
    print("  - 生成过热故障数据...")
    all_data.extend(generate_overheat_data(150, current_time))
    current_time += timedelta(minutes=150)
    
    # 不平衡故障 (150 条)
    print("  - 生成不平衡故障数据...")
    all_data.extend(generate_imbalance_data(150, current_time))
    
    # 转换为 DataFrame
    df = pd.DataFrame(all_data)
    
    # 洗牌打乱（模拟真实数据混合）
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 按时间重新排序（保留时间顺序用于滑动窗口）
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 保存 CSV
    output_path = 'limix_fault_detection/data/raw/sensor_data.csv'
    df.to_csv(output_path, index=False)
    
    # 统计数据
    print(f"\n数据集生成完成!")
    print(f"  文件路径：{output_path}")
    print(f"  总样本数：{len(df)}")
    print(f"\n各类别分布:")
    print(df['fault_label'].value_counts())
    print(f"\n数据预览:")
    print(df.head(10))
    
    return df

if __name__ == '__main__':
    generate_dataset()