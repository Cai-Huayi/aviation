import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm

# 定义常量和路径
TRAIN_DATA_DIR = r"d:\Demo\aviation\train_data\train_data"
TEST_DATA_DIR = r"d:\Demo\aviation\test_data\test_data"
FACT_DATA_DIR = os.path.join(TRAIN_DATA_DIR, "fact_data")
NWP_TRAIN_DIR = os.path.join(TRAIN_DATA_DIR, "nwp_data_train")
NWP_TEST_DIR = os.path.join(TEST_DATA_DIR, "nwp_data_test")

# 风电场站ID列表
WIND_STATION_IDS = [1, 2, 3, 4, 5]
# 光伏场站ID列表
SOLAR_STATION_IDS = [6, 7, 8, 9, 10]
# 所有场站ID列表
ALL_STATION_IDS = WIND_STATION_IDS + SOLAR_STATION_IDS

def read_power_data(station_id):
    """
    读取指定场站的功率数据
    
    参数:
    station_id: 场站ID (1-10)
    
    返回:
    pandas.DataFrame: 包含时间和功率的DataFrame
    """
    file_path = os.path.join(FACT_DATA_DIR, f"{station_id}_normalization_train.csv")
    try:
        power_data = pd.read_csv(file_path, parse_dates=['时间'])
        power_data.rename(columns={'时间': 'datetime', '功率(MW)': 'power'}, inplace=True)
        return power_data
    except Exception as e:
        print(f"读取功率数据时出错: {e}")
        return None

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
    
    try:
        # 使用xarray打开nc文件
        ds = xr.open_dataset(file_path)
        return ds
    except Exception as e:
        print(f"读取气象数据时出错 ({file_path}): {e}")
        return None

def convert_nwp_to_dataframe(ds, nwp_source, center_lat_idx=5, center_lon_idx=5):
    """
    将xarray.Dataset转换为pandas.DataFrame，只处理中心点的数据
    
    参数:
    ds: xarray.Dataset 对象
    nwp_source: 气象源，'NWP_1', 'NWP_2', 或 'NWP_3'
    center_lat_idx: 中心纬度索引，默认为5
    center_lon_idx: 中心经度索引，默认为5
    
    返回:
    pandas.DataFrame: 包含气象数据的DataFrame
    """
    if ds is None:
        return None
    
    # 确定变量列表
    if nwp_source in ['NWP_1', 'NWP_3']:
        variables = ['u100', 'v100', 't2m', 'tp', 'tcc', 'sp', 'poai', 'ghi']
    else:  # NWP_2
        variables = ['u100', 'v100', 't2m', 'tp', 'tcc', 'msl', 'poai', 'ghi']
    
    # 创建一个空的DataFrame来存储结果
    df_list = []
    
    # 获取时间点
    base_time = pd.to_datetime(ds.time.values[0])
    
    # 检查数据集中的变量和维度
    print(f"数据集维度: {ds.dims}")
    print(f"数据集变量: {list(ds.variables)}")
    
    # 检查是否有data变量和必要的维度
    if 'data' in ds.variables and 'lead_time' in ds.dims and 'channel' in ds.dims:
        print("使用data变量处理数据")
        
        # 确保中心点索引在有效范围内
        if center_lat_idx >= ds.dims['lat']:
            center_lat_idx = ds.dims['lat'] - 1
        if center_lon_idx >= ds.dims['lon']:
            center_lon_idx = ds.dims['lon'] - 1
        
        # 对每个预报时间点，提取所有变量的数据
        for hour in range(ds.dims['lead_time']):
            # 创建时间戳
            timestamp = base_time + pd.Timedelta(hours=hour)
            
            # 对于每个变量，提取数据
            for i, var_name in enumerate(variables):
                if i < ds.dims['channel']:  # 确保channel索引有效
                    try:
                        # 选择特定时间点、预报时间和变量的数据
                        data = ds.data.isel(time=0, channel=i, lead_time=hour)
                        
                        # 只处理中心点的数据
                        value = float(data.isel(lat=center_lat_idx, lon=center_lon_idx).values)
                        
                        # 创建一行数据
                        row = {
                            'datetime': timestamp,
                            'hour': hour,
                            'lat_idx': center_lat_idx,
                            'lon_idx': center_lon_idx,
                            'variable': var_name,
                            'value': value,
                            'nwp_source': nwp_source
                        }
                        
                        df_list.append(row)
                    except Exception as e:
                        print(f"处理变量 {var_name} (channel={i}) 时出错: {e}")
    else:
        print("未找到必要的变量或维度，尝试直接处理各个变量")
        
        # 确保中心点索引在有效范围内
        if center_lat_idx >= ds.dims['lat']:
            center_lat_idx = ds.dims['lat'] - 1
        if center_lon_idx >= ds.dims['lon']:
            center_lon_idx = ds.dims['lon'] - 1
        
        # 尝试直接处理各个变量
        for var_name in variables:
            if var_name in ds.variables:
                # 检查变量的维度
                var_dims = ds[var_name].dims
                print(f"变量 {var_name} 的维度: {var_dims}")
                
                # 确定时间维度
                time_dim = None
                for dim in ['lead_time', 'step', 'forecast_time']:
                    if dim in var_dims:
                        time_dim = dim
                        break
                
                if time_dim is not None:
                    # 对每个预报时间点，提取数据
                    for hour in range(ds.dims[time_dim]):
                        # 创建时间戳
                        timestamp = base_time + pd.Timedelta(hours=hour)
                        
                        try:
                            # 选择特定时间点和预报时间的数据
                            data = ds[var_name].isel(time=0, **{time_dim: hour})
                            
                            # 只处理中心点的数据
                            value = float(data.isel(lat=center_lat_idx, lon=center_lon_idx).values)
                            
                            # 创建一行数据
                            row = {
                                'datetime': timestamp,
                                'hour': hour,
                                'lat_idx': center_lat_idx,
                                'lon_idx': center_lon_idx,
                                'variable': var_name,
                                'value': value,
                                'nwp_source': nwp_source
                            }
                            
                            df_list.append(row)
                        except Exception as e:
                            print(f"处理变量 {var_name} 在时间 {hour} 时出错: {e}")
    
    # 将所有行合并为一个DataFrame
    if df_list:
        df = pd.DataFrame(df_list)
        print(f"成功创建DataFrame，形状: {df.shape}")
        return df
    else:
        print("未能创建DataFrame，数据列表为空")
        return None

def process_station_data(station_id, start_date, end_date, nwp_sources=['NWP_1', 'NWP_2', 'NWP_3']):
    """
    处理指定场站在指定日期范围内的数据
    
    参数:
    station_id: 场站ID (1-10)
    start_date: 开始日期，格式为'YYYYMMDD'
    end_date: 结束日期，格式为'YYYYMMDD'
    nwp_sources: 气象源列表
    
    返回:
    tuple: (power_df, weather_df)，分别为功率数据和气象数据
    """
    # 读取功率数据
    power_df = read_power_data(station_id)
    
    if power_df is None:
        print(f"无法读取场站 {station_id} 的功率数据")
        return None, None
    
    # 转换日期格式
    start_date_dt = datetime.strptime(start_date, '%Y%m%d')
    end_date_dt = datetime.strptime(end_date, '%Y%m%d')
    
    # 筛选日期范围内的功率数据
    power_df = power_df[(power_df['datetime'] >= start_date_dt) & 
                         (power_df['datetime'] <= end_date_dt + timedelta(days=1))]
    
    # 创建一个空的列表来存储所有气象数据
    weather_dfs = []
    
    # 生成日期范围内的所有日期
    current_date = start_date_dt
    while current_date <= end_date_dt:
        date_str = current_date.strftime('%Y%m%d')
        
        # 对每个气象源，读取并处理数据
        for nwp_source in nwp_sources:
            print(f"处理场站 {station_id} 的 {date_str} {nwp_source} 数据")
            ds = read_nwp_data(station_id, date_str, nwp_source)
            
            if ds is not None:
                # 转换为DataFrame，只处理中心点的数据
                weather_df = convert_nwp_to_dataframe(ds, nwp_source)
                
                if weather_df is not None:
                    # 添加日期信息
                    weather_df['forecast_date'] = date_str
                    
                    # 添加到列表
                    weather_dfs.append(weather_df)
                
                # 关闭数据集
                ds.close()
        
        # 移动到下一天
        current_date += timedelta(days=1)
    
    # 合并所有气象数据
    if weather_dfs:
        all_weather_df = pd.concat(weather_dfs, ignore_index=True)
        return power_df, all_weather_df
    else:
        return power_df, None

def align_power_and_weather(power_df, weather_df, center_lat_idx=5, center_lon_idx=5):
    """
    将功率数据和气象数据对齐
    
    参数:
    power_df: 功率数据DataFrame
    weather_df: 气象数据DataFrame
    center_lat_idx: 中心纬度索引
    center_lon_idx: 中心经度索引
    
    返回:
    pandas.DataFrame: 对齐后的DataFrame
    """
    if power_df is None or weather_df is None:
        print("功率数据或气象数据为空")
        return None
    
    # 创建一个新的DataFrame来存储结果
    aligned_data = []
    
    # 对于每个功率数据点
    for _, power_row in tqdm(power_df.iterrows(), total=len(power_df), desc="对齐数据"):
        power_time = power_row['datetime']
        power_value = power_row['power']
        
        # 找到对应的气象预报数据
        # 气象预报是前一天发布的，预报未来24小时
        forecast_date = (power_time - timedelta(days=1)).strftime('%Y%m%d')
        
        # 计算预报小时
        forecast_hour = power_time.hour
        
        # 筛选对应的气象数据
        forecast_weather = weather_df[
            (weather_df['forecast_date'] == forecast_date) & 
            (weather_df['hour'] == forecast_hour) &
            (weather_df['lat_idx'] == center_lat_idx) & 
            (weather_df['lon_idx'] == center_lon_idx)
        ]
        
        # 如果找到了对应的气象数据
        if not forecast_weather.empty:
            # 创建一个字典来存储这个时间点的所有数据
            data_point = {
                'datetime': power_time,
                'power': power_value,
                'forecast_date': forecast_date,
                'forecast_hour': forecast_hour
            }
            
            # 对于每个气象源
            for nwp_source in forecast_weather['nwp_source'].unique():
                source_data = forecast_weather[forecast_weather['nwp_source'] == nwp_source]
                
                # 对于每个变量
                for var_name in source_data['variable'].unique():
                    var_data = source_data[source_data['variable'] == var_name]
                    
                    # 获取值
                    if not var_data.empty:
                        data_point[f'{nwp_source}_{var_name}'] = var_data['value'].values[0]
            
            aligned_data.append(data_point)
    
    # 将所有数据点合并为一个DataFrame
    aligned_df = pd.DataFrame(aligned_data)
    
    return aligned_df

def create_features(df, is_wind_station=True):
    """
    创建特征
    
    参数:
    df: 对齐后的DataFrame
    is_wind_station: 是否为风电场站
    
    返回:
    pandas.DataFrame: 包含特征的DataFrame
    """
    if df is None or df.empty:
        print("输入数据为空")
        return None
    
    # 创建一个副本，避免修改原始数据
    features_df = df.copy()
    
    # 添加时间特征
    features_df['hour'] = features_df['datetime'].dt.hour
    features_df['day'] = features_df['datetime'].dt.day
    features_df['month'] = features_df['datetime'].dt.month
    features_df['dayofweek'] = features_df['datetime'].dt.dayofweek
    features_df['is_weekend'] = features_df['dayofweek'].isin([5, 6]).astype(int)
    
    # 添加风速特征 (对于风电场站)
    if is_wind_station:
        for nwp_source in ['NWP_1', 'NWP_2', 'NWP_3']:
            if f'{nwp_source}_u100' in features_df.columns and f'{nwp_source}_v100' in features_df.columns:
                features_df[f'{nwp_source}_wind_speed'] = np.sqrt(
                    features_df[f'{nwp_source}_u100']**2 + 
                    features_df[f'{nwp_source}_v100']**2
                )
                features_df[f'{nwp_source}_wind_direction'] = np.arctan2(
                    features_df[f'{nwp_source}_v100'], 
                    features_df[f'{nwp_source}_u100']
                )
    
    # 添加滞后特征 (前1小时、前2小时、前3小时、前4小时、前24小时的功率)
    for lag in [1, 2, 3, 4, 24]:
        features_df[f'power_lag_{lag}'] = features_df['power'].shift(lag * 4)  # 15分钟 * 4 = 1小时
    
    # 添加移动平均特征
    for window in [4, 12, 24, 96]:  # 1小时、3小时、6小时、24小时
        features_df[f'power_rolling_mean_{window}'] = features_df['power'].rolling(window=window).mean()
    
    # 添加一天中的正弦和余弦时间特征，捕捉周期性
    features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour'] / 24)
    features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour'] / 24)
    
    # 添加一年中的正弦和余弦时间特征，捕捉季节性
    features_df['day_of_year'] = features_df['datetime'].dt.dayofyear
    features_df['day_sin'] = np.sin(2 * np.pi * features_df['day_of_year'] / 365)
    features_df['day_cos'] = np.cos(2 * np.pi * features_df['day_of_year'] / 365)
    
    # 处理缺失值
    features_df = features_df.dropna()
    
    return features_df

def prepare_train_test_data(station_id, train_start_date, train_end_date, test_start_date, test_end_date):
    """
    准备训练和测试数据
    
    参数:
    station_id: 场站ID (1-10)
    train_start_date: 训练数据开始日期，格式为'YYYYMMDD'
    train_end_date: 训练数据结束日期，格式为'YYYYMMDD'
    test_start_date: 测试数据开始日期，格式为'YYYYMMDD'
    test_end_date: 测试数据结束日期，格式为'YYYYMMDD'
    
    返回:
    tuple: (train_features, train_target, test_features, test_target)
    """
    # 判断是否为风电场站
    is_wind_station = station_id in WIND_STATION_IDS
    
    # 处理训练数据
    train_power_df, train_weather_df = process_station_data(
        station_id, train_start_date, train_end_date
    )
    
    if train_power_df is None or train_weather_df is None:
        print(f"无法处理场站 {station_id} 的训练数据")
        return None, None, None, None
    
    # 对齐训练数据
    train_aligned_df = align_power_and_weather(train_power_df, train_weather_df)
    
    if train_aligned_df is None or train_aligned_df.empty:
        print(f"无法对齐场站 {station_id} 的训练数据")
        return None, None, None, None
    
    # 创建训练特征
    train_features_df = create_features(train_aligned_df, is_wind_station)
    
    if train_features_df is None or train_features_df.empty:
        print(f"无法创建场站 {station_id} 的训练特征")
        return None, None, None, None
    
    # 处理测试数据
    test_power_df, test_weather_df = process_station_data(
        station_id, test_start_date, test_end_date
    )
    
    if test_power_df is None or test_weather_df is None:
        print(f"无法处理场站 {station_id} 的测试数据")
        return train_features_df, None, None, None
    
    # 对齐测试数据
    test_aligned_df = align_power_and_weather(test_power_df, test_weather_df)
    
    if test_aligned_df is None or test_aligned_df.empty:
        print(f"无法对齐场站 {station_id} 的测试数据")
        return train_features_df, None, None, None
    
    # 创建测试特征
    test_features_df = create_features(test_aligned_df, is_wind_station)
    
    if test_features_df is None or test_features_df.empty:
        print(f"无法创建场站 {station_id} 的测试特征")
        return train_features_df, None, None, None
    
    # 分离特征和目标
    train_target = train_features_df['power']
    train_features = train_features_df.drop(['power', 'datetime'], axis=1)
    
    test_target = test_features_df['power']
    test_features = test_features_df.drop(['power', 'datetime'], axis=1)
    
    return train_features, train_target, test_features, test_target

def main():
    """
    主函数，用于测试数据处理功能
    """
    # 设置要处理的数据
    station_id = 1  # 可以是1-10
    start_date = '20240101'  # 格式为YYYYMMDD
    end_date = '20240101'  # 格式为YYYYMMDD，缩短范围以加快处理速度
    
    # 处理数据
    power_df, weather_df = process_station_data(station_id, start_date, end_date, nwp_sources=['NWP_1'])
    
    if power_df is not None:
        print(f"功率数据形状: {power_df.shape}")
        print("功率数据前5行:")
        print(power_df.head())
    else:
        print("未能读取功率数据")
    
    if weather_df is not None:
        print(f"气象数据形状: {weather_df.shape}")
        print("气象数据前5行:")
        print(weather_df.head())
        
        # 对齐数据
        aligned_df = align_power_and_weather(power_df, weather_df)
        
        if aligned_df is not None:
            print(f"对齐后数据形状: {aligned_df.shape}")
            print("对齐后数据列:")
            print(aligned_df.columns.tolist())
            
            # 创建特征
            features_df = create_features(aligned_df, is_wind_station=(station_id in WIND_STATION_IDS))
            
            if features_df is not None:
                print(f"特征数据形状: {features_df.shape}")
                print("特征数据列:")
                print(features_df.columns.tolist())
                
                # 保存样本数据
                features_df.to_csv(f"station_{station_id}_features_sample.csv", index=False)
                print(f"已保存特征样本数据到 station_{station_id}_features_sample.csv")
    else:
        print("未能读取气象数据")

if __name__ == "__main__":
    main()
