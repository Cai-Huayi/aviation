import os
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 定义路径
TRAIN_DATA_DIR = r"d:\Demo\aviation\train_data\train_data"
NWP_TRAIN_DIR = os.path.join(TRAIN_DATA_DIR, "nwp_data_train")

def debug_nc_file(station_id, date_str, nwp_source):
    """
    详细调试NC文件的结构和内容
    
    参数:
    station_id: 场站ID (1-10)
    date_str: 日期字符串，格式为'YYYYMMDD'
    nwp_source: 气象源，'NWP_1', 'NWP_2', 或 'NWP_3'
    """
    file_path = os.path.join(NWP_TRAIN_DIR, str(station_id), nwp_source, f"{date_str}.nc")
    print(f"检查文件: {file_path}")
    print(f"文件是否存在: {os.path.exists(file_path)}")
    
    try:
        # 使用xarray打开nc文件
        ds = xr.open_dataset(file_path)
        
        # 打印数据集基本信息
        print("\n数据集基本信息:")
        print(f"维度: {dict(ds.dims)}")
        print(f"坐标: {list(ds.coords)}")
        print(f"数据变量: {list(ds.data_vars)}")
        
        # 检查数据变量的结构
        print("\n数据变量结构:")
        for var_name in ds.data_vars:
            print(f"\n变量: {var_name}")
            print(f"  维度: {ds[var_name].dims}")
            print(f"  形状: {ds[var_name].shape}")
            
            # 尝试获取一个数据点
            try:
                if 'data' == var_name:
                    # 对于data变量，我们需要特殊处理
                    print("  data变量的结构:")
                    print(f"    维度: {ds.data.dims}")
                    print(f"    形状: {ds.data.shape}")
                    
                    # 检查channel维度是否存在
                    if 'channel' in ds.data.dims:
                        print("    channel维度的值:")
                        for i in range(min(8, ds.dims['channel'])):
                            try:
                                # 尝试获取每个channel的第一个数据点
                                sample = ds.data.isel(time=0, channel=i, lat=0, lon=0)
                                print(f"      channel={i}: {sample.values}")
                            except Exception as e:
                                print(f"      无法获取channel={i}的数据: {e}")
            except Exception as e:
                print(f"  无法获取数据点: {e}")
        
        # 如果存在data变量，尝试详细分析
        if 'data' in ds.data_vars:
            print("\n详细分析data变量:")
            data_var = ds.data
            
            # 检查维度
            print(f"data变量的维度: {data_var.dims}")
            
            # 检查channel维度的大小
            if 'channel' in data_var.dims:
                channel_size = data_var.dims['channel']
                print(f"channel维度的大小: {channel_size}")
                
                # 确定变量列表
                if nwp_source in ['NWP_1', 'NWP_3']:
                    variables = ['u100', 'v100', 't2m', 'tp', 'tcc', 'sp', 'poai', 'ghi']
                else:  # NWP_2
                    variables = ['u100', 'v100', 't2m', 'tp', 'tcc', 'msl', 'poai', 'ghi']
                
                print(f"预期的变量列表: {variables}")
                
                # 检查channel数量是否与变量数量匹配
                if channel_size == len(variables):
                    print("channel数量与变量数量匹配")
                    
                    # 尝试提取第一个时间点的所有channel数据
                    print("\n尝试提取第一个时间点的所有channel数据:")
                    for i, var_name in enumerate(variables):
                        try:
                            # 获取该channel的数据
                            channel_data = data_var.isel(time=0, channel=i)
                            
                            # 计算统计信息
                            min_val = float(channel_data.min().values)
                            max_val = float(channel_data.max().values)
                            mean_val = float(channel_data.mean().values)
                            
                            print(f"  {var_name} (channel={i}): 最小值={min_val}, 最大值={max_val}, 平均值={mean_val}")
                            
                            # 可视化第一个channel的数据
                            if i == 0:
                                plt.figure(figsize=(10, 6))
                                channel_data.plot()
                                plt.title(f"{var_name} 数据可视化")
                                plt.savefig(f"{var_name}_visualization.png")
                                plt.close()
                                print(f"  已保存 {var_name} 的可视化图像到 {var_name}_visualization.png")
                        except Exception as e:
                            print(f"  无法处理 {var_name} (channel={i}): {e}")
                else:
                    print(f"警告: channel数量 ({channel_size}) 与变量数量 ({len(variables)}) 不匹配")
            
            # 检查lead_time维度
            if 'lead_time' in data_var.dims:
                lead_time_size = data_var.dims['lead_time']
                print(f"\nlead_time维度的大小: {lead_time_size}")
                
                # 尝试提取不同lead_time的数据
                print("尝试提取不同lead_time的数据:")
                for i in range(min(5, lead_time_size)):
                    try:
                        # 获取该lead_time的数据
                        lead_time_data = data_var.isel(time=0, channel=0, lead_time=i)
                        
                        # 计算统计信息
                        min_val = float(lead_time_data.min().values)
                        max_val = float(lead_time_data.max().values)
                        mean_val = float(lead_time_data.mean().values)
                        
                        print(f"  lead_time={i}: 最小值={min_val}, 最大值={max_val}, 平均值={mean_val}")
                    except Exception as e:
                        print(f"  无法处理 lead_time={i}: {e}")
        
        # 尝试转换为DataFrame
        print("\n尝试转换为DataFrame:")
        try:
            # 确定变量列表
            if nwp_source in ['NWP_1', 'NWP_3']:
                variables = ['u100', 'v100', 't2m', 'tp', 'tcc', 'sp', 'poai', 'ghi']
            else:  # NWP_2
                variables = ['u100', 'v100', 't2m', 'tp', 'tcc', 'msl', 'poai', 'ghi']
            
            # 创建一个空的DataFrame来存储结果
            df_list = []
            
            # 获取时间点
            base_time = pd.to_datetime(ds.time.values[0])
            print(f"基准时间: {base_time}")
            
            # 如果存在data变量和channel维度
            if 'data' in ds.data_vars and 'channel' in ds.data.dims:
                print("使用data变量和channel维度提取数据")
                
                # 检查lead_time维度
                if 'lead_time' in ds.data.dims:
                    print(f"使用lead_time维度，大小: {ds.data.dims['lead_time']}")
                    
                    # 对每个预报时间点，提取所有变量的数据
                    for hour in range(min(5, ds.data.dims['lead_time'])):
                        # 创建时间戳
                        timestamp = base_time + pd.Timedelta(hours=hour)
                        print(f"处理预报时间: {timestamp}")
                        
                        # 对于每个变量，提取数据
                        for i, var_name in enumerate(variables):
                            try:
                                # 选择特定时间点、预报时间和变量的数据
                                data = ds.data.isel(time=0, channel=i, lead_time=hour)
                                
                                # 对于第一个lat, lon位置，创建一行数据
                                value = data.isel(lat=0, lon=0).values
                                
                                # 创建一行数据
                                row = {
                                    'datetime': timestamp,
                                    'hour': hour,
                                    'lat_idx': 0,
                                    'lon_idx': 0,
                                    'variable': var_name,
                                    'value': float(value),
                                    'nwp_source': nwp_source
                                }
                                
                                df_list.append(row)
                                print(f"  成功提取 {var_name} 的数据")
                            except Exception as e:
                                print(f"  提取 {var_name} 的数据时出错: {e}")
                else:
                    print("未找到lead_time维度")
            else:
                print("未找到data变量或channel维度")
            
            # 将所有行合并为一个DataFrame
            if df_list:
                df = pd.DataFrame(df_list)
                print("\n成功创建DataFrame:")
                print(df.head())
            else:
                print("未能创建DataFrame，数据列表为空")
        except Exception as e:
            print(f"转换为DataFrame时出错: {e}")
        
        return ds
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

def main():
    # 检查第一个场站的NWP_1数据
    station_id = 1
    date_str = '20240101'
    nwp_source = 'NWP_1'
    
    ds = debug_nc_file(station_id, date_str, nwp_source)
    
    if ds is not None:
        # 关闭数据集
        ds.close()
        print("\n成功关闭数据集")

if __name__ == "__main__":
    main()
