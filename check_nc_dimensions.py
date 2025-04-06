import os
import xarray as xr
import numpy as np
import pandas as pd

# 定义路径
TRAIN_DATA_DIR = r"d:\Demo\aviation\train_data\train_data"
NWP_TRAIN_DIR = os.path.join(TRAIN_DATA_DIR, "nwp_data_train")

def check_nc_file(station_id, date_str, nwp_source):
    """
    检查NC文件的维度和变量
    
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
        
        # 打印数据集信息
        print("\n数据集信息:")
        print(ds)
        
        # 打印维度信息
        print("\n维度信息:")
        for dim_name, dim_size in ds.dims.items():
            print(f"  {dim_name}: {dim_size}")
        
        # 打印坐标信息
        print("\n坐标信息:")
        for coord_name, coord in ds.coords.items():
            print(f"  {coord_name}: {coord.dims}, 形状: {coord.shape}")
            if len(coord.values) < 10:  # 只打印少量值，避免输出过多
                print(f"    值: {coord.values}")
            else:
                print(f"    前5个值: {coord.values[:5]}")
                print(f"    后5个值: {coord.values[-5:]}")
        
        # 打印变量信息
        print("\n变量信息:")
        for var_name, var in ds.variables.items():
            if var_name not in ds.dims and var_name not in ds.coords:
                print(f"  {var_name}: {var.dims}, 形状: {var.shape}")
                
                # 尝试获取一个数据点作为示例
                try:
                    if len(var.shape) > 2:  # 多维数组
                        sample = var.isel(time=0, step=0, lat=0, lon=0).values
                        print(f"    示例值 (time=0, step=0, lat=0, lon=0): {sample}")
                    elif len(var.shape) == 2:  # 二维数组
                        sample = var.isel(lat=0, lon=0).values
                        print(f"    示例值 (lat=0, lon=0): {sample}")
                    elif len(var.shape) == 1:  # 一维数组
                        sample = var.values[0]
                        print(f"    示例值 (索引0): {sample}")
                except Exception as e:
                    print(f"    无法获取示例值: {e}")
        
        # 尝试提取数据并转换为DataFrame
        print("\n尝试提取数据并转换为DataFrame:")
        try:
            # 获取第一个变量
            first_var = next(var_name for var_name in ds.variables if var_name not in ds.dims and var_name not in ds.coords)
            
            # 打印变量的维度名称
            print(f"变量 {first_var} 的维度: {ds[first_var].dims}")
            
            # 尝试提取数据
            if 'step' in ds[first_var].dims:
                print("使用 'step' 作为预报时间维度")
                data = ds[first_var].isel(time=0, step=0)
                print(f"提取的数据形状: {data.shape}")
            elif 'lead_time' in ds[first_var].dims:
                print("使用 'lead_time' 作为预报时间维度")
                data = ds[first_var].isel(time=0, lead_time=0)
                print(f"提取的数据形状: {data.shape}")
            else:
                print("未找到预报时间维度")
        except Exception as e:
            print(f"提取数据时出错: {e}")
        
        return ds
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

def main():
    # 检查第一个场站的NWP_1数据
    station_id = 1
    date_str = '20240101'
    nwp_source = 'NWP_1'
    
    ds = check_nc_file(station_id, date_str, nwp_source)
    
    if ds is not None:
        # 检查变量的维度
        print("\n变量的维度详情:")
        for var_name in ds.variables:
            if var_name not in ds.dims and var_name not in ds.coords:
                print(f"  {var_name}: {ds[var_name].dims}")
        
        # 尝试提取数据并转换为pandas DataFrame
        print("\n尝试转换为DataFrame:")
        try:
            # 获取第一个变量名
            first_var = next(var_name for var_name in ds.variables if var_name not in ds.dims and var_name not in ds.coords)
            
            # 打印变量的维度名称
            print(f"变量 {first_var} 的维度: {ds[first_var].dims}")
            
            # 尝试不同的维度名称
            possible_time_dims = ['step', 'lead_time', 'forecast_time']
            
            for time_dim in possible_time_dims:
                if time_dim in ds[first_var].dims:
                    print(f"找到时间维度: {time_dim}")
                    # 获取时间点
                    base_time = pd.to_datetime(ds.time.values[0])
                    print(f"基准时间: {base_time}")
                    
                    # 创建一个空的DataFrame来存储结果
                    df_list = []
                    
                    # 对第一个小时，提取第一个变量的数据
                    data = ds[first_var].isel(time=0, **{time_dim: 0})
                    
                    # 对于第一个lat, lon位置，创建一行数据
                    value = data.isel(lat=0, lon=0).values
                    
                    # 创建一行数据
                    row = {
                        'datetime': base_time,
                        time_dim: 0,
                        'lat_idx': 0,
                        'lon_idx': 0,
                        'variable': first_var,
                        'value': float(value),
                        'nwp_source': nwp_source
                    }
                    
                    df_list.append(row)
                    
                    # 将所有行合并为一个DataFrame
                    df = pd.DataFrame(df_list)
                    
                    print("成功创建DataFrame:")
                    print(df)
                    break
            else:
                print("未找到时间维度")
        except Exception as e:
            print(f"转换为DataFrame时出错: {e}")

if __name__ == "__main__":
    main()
