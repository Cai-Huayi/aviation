import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 路径设置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DATA_DIR = os.path.join(BASE_DIR, 'train_data', 'train_data')
TEST_DATA_DIR = os.path.join(BASE_DIR, 'test_data', 'test_data')
FACT_DATA_DIR = os.path.join(TRAIN_DATA_DIR, 'fact_data')
NWP_TRAIN_DIR = os.path.join(TRAIN_DATA_DIR, 'nwp_data_train')
NWP_TEST_DIR = os.path.join(TEST_DATA_DIR, 'nwp_data_test')

# 函数：读取功率数据
def read_power_data(station_id):
    """读取指定场站的功率数据"""
    file_path = os.path.join(FACT_DATA_DIR, f'{station_id}_normalization_train.csv')
    power_data = pd.read_csv(file_path, parse_dates=['时间'])
    power_data.rename(columns={'时间': 'datetime', '功率(MW)': 'power'}, inplace=True)
    return power_data

# 函数：读取气象数据
def read_nwp_data(station_id, date_str, nwp_source, is_test=False):
    """
    读取指定场站、指定日期、指定气象源的气象数据
    
    参数:
    station_id: 场站ID (1-10)
    date_str: 日期字符串，格式为'YYYYMMDD'
    nwp_source: 气象源，'NWP_1', 'NWP_2', 或 'NWP_3'
    is_test: 是否为测试数据
    
    返回:
    xarray.Dataset: 包含气象数据的Dataset
    """
    if is_test:
        base_dir = NWP_TEST_DIR
    else:
        base_dir = NWP_TRAIN_DIR
    
    file_path = os.path.join(base_dir, str(station_id), nwp_source, f'{date_str}.nc')
    
    try:
        # 使用xarray打开nc文件
        ds = xr.open_dataset(file_path)
        return ds
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return None

# 函数：探索功率数据
def explore_power_data(station_id):
    """探索指定场站的功率数据"""
    power_data = read_power_data(station_id)
    
    print(f"\n===== 场站 {station_id} 功率数据 =====")
    print(f"数据形状: {power_data.shape}")
    print(f"数据时间范围: {power_data['datetime'].min()} 到 {power_data['datetime'].max()}")
    print(f"数据采样间隔: {(power_data['datetime'].iloc[1] - power_data['datetime'].iloc[0]).total_seconds() / 60} 分钟")
    print(f"功率统计信息:")
    print(power_data['power'].describe())
    
    # 检查缺失值
    missing_values = power_data.isnull().sum()
    print(f"缺失值数量: {missing_values}")
    
    # 检查异常值 (简单方法：超过3个标准差的值)
    mean = power_data['power'].mean()
    std = power_data['power'].std()
    outliers = power_data[(power_data['power'] > mean + 3*std) | (power_data['power'] < mean - 3*std)]
    print(f"潜在异常值数量: {len(outliers)}")
    
    # 绘制功率时间序列图
    plt.figure(figsize=(15, 6))
    plt.plot(power_data['datetime'], power_data['power'])
    plt.title(f'场站 {station_id} 功率时间序列')
    plt.xlabel('时间')
    plt.ylabel('归一化功率 (MW)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'station_{station_id}_power_timeseries.png')
    plt.close()
    
    return power_data

# 函数：探索气象数据
def explore_nwp_data(station_id, date_str, nwp_source, is_test=False):
    """探索指定场站、指定日期、指定气象源的气象数据"""
    ds = read_nwp_data(station_id, date_str, nwp_source, is_test)
    
    if ds is None:
        return None
    
    print(f"\n===== 场站 {station_id}, 日期 {date_str}, 气象源 {nwp_source} =====")
    print(f"数据维度:")
    for dim_name, dim_size in ds.dims.items():
        print(f"  {dim_name}: {dim_size}")
    
    print(f"数据变量:")
    for var_name, var in ds.variables.items():
        if var_name not in ds.dims:
            print(f"  {var_name}: {var.dims}, 形状: {var.shape}")
    
    # 打印变量的基本统计信息
    for var_name, var in ds.variables.items():
        if var_name not in ds.dims and len(var.shape) > 0:
            try:
                print(f"\n{var_name} 统计信息:")
                data = var.values
                print(f"  最小值: {np.nanmin(data)}")
                print(f"  最大值: {np.nanmax(data)}")
                print(f"  平均值: {np.nanmean(data)}")
                print(f"  标准差: {np.nanstd(data)}")
                print(f"  缺失值数量: {np.isnan(data).sum()}")
            except Exception as e:
                print(f"  计算 {var_name} 统计信息时出错: {e}")
    
    return ds

# 主函数
def main():
    # 探索功率数据
    for station_id in range(1, 11):
        power_data = explore_power_data(station_id)
    
    # 探索气象数据 (以第一个场站的一天数据为例)
    station_id = 1
    date_str = '20240101'
    
    for nwp_source in ['NWP_1', 'NWP_2', 'NWP_3']:
        ds = explore_nwp_data(station_id, date_str, nwp_source)
        
        if ds is not None:
            # 可视化一些气象变量
            if nwp_source in ['NWP_1', 'NWP_3']:
                variables = ['u100', 'v100', 't2m', 'tp', 'tcc', 'sp', 'poai', 'ghi']
            else:  # NWP_2
                variables = ['u100', 'v100', 't2m', 'tp', 'tcc', 'msl', 'poai', 'ghi']
            
            for var_name in variables:
                if var_name in ds:
                    try:
                        # 选择第一个时间点和通道
                        data = ds[var_name].isel(time=0, channel=0)
                        
                        plt.figure(figsize=(10, 8))
                        # 绘制热力图
                        plt.imshow(data.values, cmap='viridis')
                        plt.colorbar(label=var_name)
                        plt.title(f'场站 {station_id}, {nwp_source}, {var_name}, 时间点 0')
                        plt.savefig(f'station_{station_id}_{nwp_source}_{var_name}_heatmap.png')
                        plt.close()
                    except Exception as e:
                        print(f"绘制 {var_name} 热力图时出错: {e}")

if __name__ == "__main__":
    main()
