import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 定义常量和路径
TRAIN_DATA_DIR = r"d:\Demo\aviation\train_data\train_data"
TEST_DATA_DIR = r"d:\Demo\aviation\test_data\test_data"
FACT_DATA_DIR = os.path.join(TRAIN_DATA_DIR, "fact_data")
NWP_TRAIN_DIR = os.path.join(TRAIN_DATA_DIR, "nwp_data_train")
NWP_TEST_DIR = os.path.join(TEST_DATA_DIR, "nwp_data_test")

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
    
    file_path = os.path.join(base_dir, str(station_id), nwp_source, f"{date_str}.nc")
    print(f"尝试读取文件: {file_path}")
    print(f"文件是否存在: {os.path.exists(file_path)}")
    
    try:
        # 使用xarray打开nc文件
        ds = xr.open_dataset(file_path)
        return ds
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

def explore_nwp_data(ds, station_id, date_str, nwp_source):
    """
    探索气象数据的结构和内容
    
    参数:
    ds: xarray.Dataset 对象
    station_id: 场站ID
    date_str: 日期字符串
    nwp_source: 气象源
    """
    if ds is None:
        print("数据集为空，无法进行探索")
        return
    
    print(f"\n===== 场站 {station_id}, 日期 {date_str}, 气象源 {nwp_source} =====")
    
    # 打印数据集基本信息
    print("\n数据集基本信息:")
    print(ds)
    
    # 打印维度信息
    print("\n维度信息:")
    for dim_name, dim_size in ds.dims.items():
        print(f"  {dim_name}: {dim_size}")
    
    # 打印变量信息
    print("\n变量信息:")
    for var_name, var in ds.variables.items():
        if var_name not in ds.dims:
            print(f"  {var_name}: {var.dims}, 形状: {var.shape}")
            
            # 打印变量的基本统计信息
            try:
                data = var.values
                print(f"    最小值: {np.nanmin(data)}")
                print(f"    最大值: {np.nanmax(data)}")
                print(f"    平均值: {np.nanmean(data)}")
                print(f"    标准差: {np.nanstd(data)}")
                print(f"    缺失值数量: {np.isnan(data).sum()}")
            except Exception as e:
                print(f"    计算统计信息时出错: {e}")

def visualize_nwp_variable(ds, var_name, station_id, date_str, nwp_source, time_idx=0, channel_idx=0, hour_idx=0):
    """
    可视化气象变量
    
    参数:
    ds: xarray.Dataset 对象
    var_name: 变量名称
    station_id: 场站ID
    date_str: 日期字符串
    nwp_source: 气象源
    time_idx: 时间索引
    channel_idx: 通道索引
    hour_idx: 小时索引
    """
    if ds is None or var_name not in ds:
        print(f"数据集为空或不包含变量 {var_name}")
        return
    
    try:
        # 选择指定的时间点、通道和小时
        data = ds[var_name].isel(time=time_idx, channel=channel_idx, hour=hour_idx)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(data.values, cmap='viridis')
        plt.colorbar(label=var_name)
        plt.title(f'站点 {station_id}, {nwp_source}, {var_name}, 时间点 {time_idx}, 小时 {hour_idx}')
        
        # 保存图像
        output_file = f'station_{station_id}_{nwp_source}_{var_name}_t{time_idx}_h{hour_idx}.png'
        plt.savefig(output_file)
        plt.close()
        print(f"已保存图像到 {output_file}")
        
    except Exception as e:
        print(f"可视化变量 {var_name} 时出错: {e}")

def visualize_time_series(ds, var_name, station_id, date_str, nwp_source, lat_idx=5, lon_idx=5):
    """
    可视化变量随时间的变化
    
    参数:
    ds: xarray.Dataset 对象
    var_name: 变量名称
    station_id: 场站ID
    date_str: 日期字符串
    nwp_source: 气象源
    lat_idx: 纬度索引
    lon_idx: 经度索引
    """
    if ds is None or var_name not in ds:
        print(f"数据集为空或不包含变量 {var_name}")
        return
    
    try:
        # 获取变量在所有小时的数据，对于特定的位置和通道
        data = ds[var_name].isel(time=0, channel=0, lat=lat_idx, lon=lon_idx)
        
        plt.figure(figsize=(12, 6))
        plt.plot(data.hour.values, data.values, marker='o')
        plt.grid(True)
        plt.xlabel('预报小时')
        plt.ylabel(var_name)
        plt.title(f'站点 {station_id}, {nwp_source}, {var_name}, 位置 ({lat_idx},{lon_idx})')
        
        # 保存图像
        output_file = f'station_{station_id}_{nwp_source}_{var_name}_timeseries.png'
        plt.savefig(output_file)
        plt.close()
        print(f"已保存时间序列图到 {output_file}")
        
    except Exception as e:
        print(f"可视化变量 {var_name} 的时间序列时出错: {e}")

def compare_nwp_sources(station_id, date_str, var_name, lat_idx=5, lon_idx=5):
    """
    比较不同气象源的同一变量
    
    参数:
    station_id: 场站ID
    date_str: 日期字符串
    var_name: 变量名称
    lat_idx: 纬度索引
    lon_idx: 经度索引
    """
    plt.figure(figsize=(12, 6))
    
    for nwp_source in ['NWP_1', 'NWP_2', 'NWP_3']:
        # 对于NWP_2，如果变量是sp，需要使用msl
        current_var = var_name
        if nwp_source == 'NWP_2' and var_name == 'sp':
            current_var = 'msl'
        
        ds = read_nwp_data(station_id, date_str, nwp_source)
        
        if ds is not None and current_var in ds:
            try:
                # 获取变量在所有小时的数据，对于特定的位置和通道
                data = ds[current_var].isel(time=0, channel=0, lat=lat_idx, lon=lon_idx)
                
                plt.plot(data.hour.values, data.values, marker='o', label=f'{nwp_source} - {current_var}')
            except Exception as e:
                print(f"处理 {nwp_source} 的 {current_var} 时出错: {e}")
    
    plt.grid(True)
    plt.xlabel('预报小时')
    plt.ylabel(var_name)
    plt.title(f'站点 {station_id}, 不同气象源的 {var_name} 比较, 位置 ({lat_idx},{lon_idx})')
    plt.legend()
    
    # 保存图像
    output_file = f'station_{station_id}_{var_name}_sources_comparison.png'
    plt.savefig(output_file)
    plt.close()
    print(f"已保存气象源比较图到 {output_file}")

def main():
    # 设置要分析的数据
    station_id = 1  # 可以是1-10
    date_str = '20240101'  # 格式为YYYYMMDD
    nwp_source = 'NWP_1'  # 可以是'NWP_1', 'NWP_2', 或 'NWP_3'
    
    # 读取气象数据
    ds = read_nwp_data(station_id, date_str, nwp_source)
    
    if ds is not None:
        # 探索数据结构
        explore_nwp_data(ds, station_id, date_str, nwp_source)
        
        # 可视化不同变量
        if nwp_source in ['NWP_1', 'NWP_3']:
            variables = ['u100', 'v100', 't2m', 'tp', 'tcc', 'sp', 'poai', 'ghi']
        else:  # NWP_2
            variables = ['u100', 'v100', 't2m', 'tp', 'tcc', 'msl', 'poai', 'ghi']
        
        for var_name in variables:
            # 可视化热力图
            visualize_nwp_variable(ds, var_name, station_id, date_str, nwp_source)
            
            # 可视化时间序列
            visualize_time_series(ds, var_name, station_id, date_str, nwp_source)
        
        # 比较不同气象源的同一变量
        for var_name in ['t2m', 'u100', 'v100', 'poai', 'ghi']:
            compare_nwp_sources(station_id, date_str, var_name)

if __name__ == "__main__":
    main()
