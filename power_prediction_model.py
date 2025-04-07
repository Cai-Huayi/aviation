import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import joblib
import xgboost as xgb
import lightgbm as lgb
from data_processor import process_station_data, align_power_and_weather, create_features, WIND_STATION_IDS, read_power_data, read_nwp_data
import argparse
import traceback

# 定义常量
MODELS_DIR = "models"
RESULTS_DIR = "results"
TRAIN_DATA_DIR = os.path.join(os.path.dirname(__file__), 'train_data')  # 更新TRAIN_DATA_DIR常量

# 确保目录存在
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def prepare_data_for_station(station_id, start_date, end_date, test_size=0.2):
    """
    准备指定场站的训练和测试数据
    
    参数:
    station_id: 场站ID (1-10)
    start_date: 开始日期，格式为'YYYYMMDD'
    end_date: 结束日期，格式为'YYYYMMDD'
    test_size: 测试集比例
    
    返回:
    tuple: (X_train, X_test, y_train, y_test)
    """
    print(f"为场站 {station_id} 准备数据...")
    
    # 处理数据
    power_df, weather_df = process_station_data(station_id, start_date, end_date)
    
    if power_df is None or weather_df is None:
        print(f"无法处理场站 {station_id} 的数据")
        return None, None, None, None
    
    # 对齐数据
    aligned_df = align_power_and_weather(power_df, weather_df)
    
    if aligned_df is None or aligned_df.empty:
        print(f"无法对齐场站 {station_id} 的数据")
        return None, None, None, None
    
    # 创建特征
    features_df = create_features(aligned_df, is_wind_station=(station_id in WIND_STATION_IDS))
    
    if features_df is None or features_df.empty:
        print(f"无法创建场站 {station_id} 的特征")
        return None, None, None, None
    
    # 分离特征和目标
    y = features_df['power']
    X = features_df.drop(['power', 'datetime', 'forecast_date', 'forecast_hour'], axis=1, errors='ignore')
    
    # 按时间顺序分割训练集和测试集
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"训练集形状: {X_train.shape}, 测试集形状: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, model_type='xgboost', station_id=None):
    """
    训练模型
    
    参数:
    X_train: 训练特征
    y_train: 训练标签
    model_type: 模型类型，可选 'xgboost', 'lightgbm', 'random_forest'
    station_id: 场站ID，用于特定场站的模型调整
    
    返回:
    训练好的模型
    """
    if model_type == 'xgboost':
        from xgboost import XGBRegressor
        
        # 根据场站类型调整模型参数
        if station_id in WIND_STATION_IDS:
            # 风电场站的模型参数
            model = XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0,
                reg_alpha=0.1,
                reg_lambda=1,
                random_state=42,
                n_jobs=-1
            )
        else:
            # 光伏场站的模型参数
            model = XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=7,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0,
                reg_alpha=0.1,
                reg_lambda=1,
                random_state=42,
                n_jobs=-1
            )
    elif model_type == 'lightgbm':
        from lightgbm import LGBMRegressor
        
        # 根据场站类型调整模型参数
        if station_id in WIND_STATION_IDS:
            # 风电场站的模型参数
            model = LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                num_leaves=63,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1,
                random_state=42,
                n_jobs=-1
            )
        else:
            # 光伏场站的模型参数
            model = LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=7,
                num_leaves=127,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1,
                random_state=42,
                n_jobs=-1
            )
    elif model_type == 'random_forest':
        from sklearn.ensemble import RandomForestRegressor
        
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 训练模型
    print(f"训练 {model_type} 模型，训练集大小: {X_train.shape}")
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    评估模型性能
    
    参数:
    model: 训练好的模型
    X_test: 测试特征
    y_test: 测试目标
    
    返回:
    dict: 包含各种评估指标的字典
    """
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算评估指标
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"评估结果:")
    print(f"  MAE: {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  R²: {r2:.4f}")
    
    # 可视化预测结果
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values, label='实际值')
    plt.plot(y_pred, label='预测值')
    plt.legend()
    plt.title('预测结果对比')
    plt.xlabel('样本索引')
    plt.ylabel('功率')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_results_dir = os.path.join(RESULTS_DIR, timestamp)
    os.makedirs(current_results_dir, exist_ok=True)
    plt.savefig(os.path.join(current_results_dir, 'prediction_comparison.png'))
    plt.close()
    
    # 可视化预测误差
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values - y_pred)
    plt.title('预测误差')
    plt.xlabel('样本索引')
    plt.ylabel('误差')
    plt.savefig(os.path.join(current_results_dir, 'prediction_error.png'))
    plt.close()
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }

def predict_next_day(station_id, model, prediction_date, nwp_sources=['NWP_1', 'NWP_2', 'NWP_3']):
    """
    预测次日零时起到未来24小时逐15分钟级新能源场站发电功率
    
    参数:
    station_id: 场站ID (1-10)
    model: 训练好的模型
    prediction_date: 预测日期，格式为'YYYYMMDD'
    nwp_sources: 气象源列表
    
    返回:
    pandas.DataFrame: 包含预测结果的DataFrame
    """
    # 转换日期格式
    prediction_date_dt = datetime.strptime(prediction_date, '%Y%m%d')
    
    # 处理2025年的预测 - 使用测试数据目录中的气象数据
    if prediction_date_dt.year == 2025:
        print(f"预测日期为2025年，使用测试数据目录中的气象数据")
        # 使用测试数据目录中的气象数据
        weather_dfs = []
        for nwp_source in nwp_sources:
            try:
                # 从测试数据目录读取气象数据
                from data_processor import read_nwp_data
                ds = read_nwp_data(station_id, prediction_date, nwp_source, is_test=True)
                
                if ds is not None:
                    from data_processor import convert_nwp_to_dataframe
                    weather_df = convert_nwp_to_dataframe(ds, nwp_source)
                    if weather_df is not None:
                        weather_df['forecast_date'] = prediction_date
                        weather_dfs.append(weather_df)
                else:
                    print(f"测试气象数据文件不存在: {nwp_source}/{prediction_date}.nc")
            except Exception as e:
                print(f"读取测试气象数据时出错 ({nwp_source}/{prediction_date}.nc): {e}")
        
        if weather_dfs:
            weather_df = pd.concat(weather_dfs, ignore_index=True)
        else:
            print(f"无法获取场站 {station_id} 的气象预报数据，使用默认值生成预测结果")
            return create_default_prediction(station_id, prediction_date_dt)
    else:
        # 对于非2025年的日期，使用训练数据目录
        # 处理气象数据
        _, weather_df = process_station_data(station_id, prediction_date, prediction_date, nwp_sources=nwp_sources)
    
    if weather_df is None or weather_df.empty:
        print(f"无法获取场站 {station_id} 的气象预报数据，使用默认值生成预测结果")
        return create_default_prediction(station_id, prediction_date_dt)
    
    # 创建预测时间点
    predictions = []
    
    # 创建从预测日期0点开始的96个15分钟时间点
    pred_times = []
    for i in range(24 * 4):  # 24小时 * 每小时4个15分钟
        pred_time = prediction_date_dt + timedelta(minutes=15 * i)
        pred_times.append(pred_time)
    
    # 初始化上一个有效预测值
    last_valid_prediction = 0.0
    
    # 对每个时间点进行预测
    for pred_time in pred_times:
        # 获取对应的预报日期和小时
        forecast_date = prediction_date
        forecast_hour = pred_time.hour
        forecast_minute = pred_time.minute
        
        # 筛选对应的气象数据
        forecast_weather = weather_df[
            (weather_df['forecast_date'] == forecast_date) & 
            (weather_df['hour'] == forecast_hour)
        ]
        
        if forecast_weather.empty:
            print(f"无法找到 {pred_time} 的气象预报数据，使用上一个有效预测值")
            # 使用上一个有效预测值
            predictions.append({
                'datetime': pred_time,
                'power': last_valid_prediction
            })
            continue
        
        try:
            # 创建特征
            features = {}
            
            # 添加时间特征
            features['hour'] = pred_time.hour
            features['minute'] = pred_time.minute
            features['day'] = pred_time.day
            features['month'] = pred_time.month
            features['dayofweek'] = pred_time.weekday()
            features['is_weekend'] = int(pred_time.weekday() in [5, 6])
            
            # 添加周期性时间特征
            features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
            features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
            features['minute_sin'] = np.sin(2 * np.pi * (features['hour'] * 60 + features['minute']) / (24 * 60))
            features['minute_cos'] = np.cos(2 * np.pi * (features['hour'] * 60 + features['minute']) / (24 * 60))
            features['day_of_year'] = pred_time.timetuple().tm_yday
            features['day_sin'] = np.sin(2 * np.pi * features['day_of_year'] / 365)
            features['day_cos'] = np.cos(2 * np.pi * features['day_of_year'] / 365)
            
            # 对于每个气象源
            for nwp_source in forecast_weather['nwp_source'].unique():
                source_data = forecast_weather[forecast_weather['nwp_source'] == nwp_source]
                
                # 对于每个变量
                for var_name in source_data['variable'].unique():
                    var_data = source_data[source_data['variable'] == var_name]
                    
                    # 获取中心点的值
                    center_value = var_data[
                        (var_data['lat_idx'] == 5) & 
                        (var_data['lon_idx'] == 5)
                    ]['value'].values
                    
                    if len(center_value) > 0:
                        features[f'{nwp_source}_{var_name}'] = center_value[0]
                        
                        # 添加变量的平方和立方项，捕捉非线性关系
                        features[f'{nwp_source}_{var_name}_squared'] = center_value[0] ** 2
                        features[f'{nwp_source}_{var_name}_cubed'] = center_value[0] ** 3
            
            # 添加风速特征 (对于风电场站)
            if station_id in WIND_STATION_IDS:
                for nwp_source in nwp_sources:
                    if f'{nwp_source}_u100' in features and f'{nwp_source}_v100' in features:
                        # 计算风速
                        features[f'{nwp_source}_wind_speed'] = np.sqrt(
                            features[f'{nwp_source}_u100']**2 + 
                            features[f'{nwp_source}_v100']**2
                        )
                        # 计算风向
                        features[f'{nwp_source}_wind_direction'] = np.arctan2(
                            features[f'{nwp_source}_v100'], 
                            features[f'{nwp_source}_u100']
                        )
                        # 添加风速的平方和立方项
                        features[f'{nwp_source}_wind_speed_squared'] = features[f'{nwp_source}_wind_speed'] ** 2
                        features[f'{nwp_source}_wind_speed_cubed'] = features[f'{nwp_source}_wind_speed'] ** 3
            
            # 添加太阳辐射特征 (对于光伏场站)
            if station_id not in WIND_STATION_IDS:
                # 如果是光伏场站，添加太阳高度角特征
                day_of_year = pred_time.timetuple().tm_yday
                hour_of_day = pred_time.hour + pred_time.minute / 60.0
                
                # 简化的太阳高度角计算（这里只是一个近似）
                solar_elevation = np.sin(np.pi * day_of_year / 365) * np.sin(np.pi * hour_of_day / 24)
                features['solar_elevation'] = max(0, solar_elevation)  # 夜间为0
                
                # 添加是否白天的特征
                features['is_daytime'] = 1 if 6 <= pred_time.hour < 18 else 0
            
            # 转换为DataFrame
            features_df = pd.DataFrame([features])
            
            # 确保特征列与训练模型时使用的列一致
            # 尝试获取模型的特征名称，不同模型可能使用不同的属性名
            feature_names = None
            if hasattr(model, 'feature_names_'):
                feature_names = model.feature_names_
            elif hasattr(model, 'feature_names_in_'):
                feature_names = model.feature_names_in_
            
            if feature_names is not None:
                # 添加缺失的特征列
                missing_cols = set(feature_names) - set(features_df.columns)
                for col in missing_cols:
                    features_df[col] = 0  # 使用0填充缺失的特征
                
                # 保持列的顺序与训练时一致
                features_df = features_df[feature_names]
            
            # 预测
            power_pred = model.predict(features_df)[0]
            
            # 确保预测值在合理范围内 (0-1之间)
            power_pred = max(0, min(1, power_pred))
            
            # 根据时间调整预测值（例如，夜间光伏发电应该接近0）
            if station_id not in WIND_STATION_IDS and (pred_time.hour < 6 or pred_time.hour >= 18):
                power_pred = power_pred * 0.1  # 夜间光伏发电接近0
            
            last_valid_prediction = power_pred  # 更新上一个有效预测值
            
            # 添加到预测结果
            predictions.append({
                'datetime': pred_time,
                'power': power_pred
            })
        except Exception as e:
            print(f"预测 {pred_time} 时出错: {e}，使用上一个有效预测值")
            # 使用上一个有效预测值
            predictions.append({
                'datetime': pred_time,
                'power': last_valid_prediction
            })
    
    # 将所有预测结果合并为一个DataFrame
    predictions_df = pd.DataFrame(predictions)
    
    # 格式化日期时间为指定格式
    predictions_df['datetime'] = predictions_df['datetime'].dt.strftime('%Y/%m/%d %H:%M')
    predictions_df.set_index('datetime', inplace=True)
    
    return predictions_df

def predict_multiple_days(station_id, model, start_date, end_date, nwp_sources=['NWP_1', 'NWP_2', 'NWP_3']):
    """
    预测多天的功率
    
    参数:
    station_id: 场站ID (1-10)
    model: 训练好的模型
    start_date: 开始日期，格式为'YYYYMMDD'
    end_date: 结束日期，格式为'YYYYMMDD'
    nwp_sources: 气象源列表
    
    返回:
    pandas.DataFrame: 包含预测结果的DataFrame
    """
    # 转换日期格式
    start_date_dt = datetime.strptime(start_date, '%Y%m%d')
    end_date_dt = datetime.strptime(end_date, '%Y%m%d')
    
    # 创建日期范围
    date_range = []
    current_date = start_date_dt
    while current_date <= end_date_dt:
        date_range.append(current_date.strftime('%Y%m%d'))
        current_date += timedelta(days=1)
    
    # 创建一个空的列表来存储所有预测结果
    all_predictions = []
    
    # 对每一天进行预测
    for pred_date in date_range:
        try:
            print(f"预测场站 {station_id} 在 {pred_date} 的功率...")
            predictions = predict_next_day(
                station_id, 
                model, 
                pred_date, 
                nwp_sources
            )
            
            # 将预测结果添加到总预测结果中
            if predictions is not None:
                for idx, row in predictions.iterrows():
                    all_predictions.append({
                        'datetime': idx,
                        'power': row['power']
                    })
        except Exception as e:
            print(f"预测场站 {station_id} 在 {pred_date} 的功率时出错: {e}")
            # 创建一个默认的预测结果
            prediction_date_dt = datetime.strptime(pred_date, '%Y%m%d')
            default_df = create_default_prediction(station_id, prediction_date_dt, save_results=False)
            
            # 将默认预测结果添加到总预测结果中
            for idx, row in default_df.iterrows():
                all_predictions.append({
                    'datetime': idx,
                    'power': row['power']
                })
    
    # 将所有预测结果合并为一个DataFrame
    all_predictions_df = pd.DataFrame(all_predictions)
    
    # 格式化日期时间为指定格式
    if not all_predictions_df.empty:
        all_predictions_df.set_index('datetime', inplace=True)
        # 确保索引是按时间排序的
        all_predictions_df = all_predictions_df.sort_index()
    
    return all_predictions_df

def create_default_prediction(station_id, prediction_date_dt, save_results=True):
    """
    当无法获取气象数据时，创建默认的预测结果
    
    参数:
    station_id: 场站ID (1-10)
    prediction_date_dt: 预测日期的datetime对象
    save_results: 是否保存结果
    
    返回:
    pandas.DataFrame: 包含默认预测结果的DataFrame
    """
    print(f"为场站 {station_id} 创建默认预测结果")
    
    # 创建预测时间点
    prediction_times = []
    for i in range(24 * 4):  # 24小时 * 每小时4个15分钟
        prediction_times.append(prediction_date_dt + timedelta(minutes=15 * i))
    
    # 获取基于历史数据的默认值
    default_values = generate_default_values_from_history(station_id, prediction_date_dt)
    
    # 创建预测结果
    predictions = []
    for i, pred_time in enumerate(prediction_times):
        predictions.append({
            'datetime': pred_time,
            'power': default_values[i]
        })
    
    # 将所有预测结果合并为一个DataFrame
    predictions_df = pd.DataFrame(predictions)
    
    # 格式化日期时间为指定格式
    predictions_df['datetime'] = predictions_df['datetime'].dt.strftime('%Y/%m/%d %H:%M')
    predictions_df.set_index('datetime', inplace=True)
    
    if save_results:
        # 保存预测结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        current_results_dir = os.path.join(RESULTS_DIR, timestamp)
        os.makedirs(current_results_dir, exist_ok=True)
        output_path = os.path.join(current_results_dir, f"output{station_id}.csv")
        predictions_df.to_csv(output_path)
        print(f"默认预测结果已保存到 {output_path}")
    
    return predictions_df

def generate_default_values_from_history(station_id, pred_date):
    """
    基于历史数据的统计特性生成默认预测值
    
    参数:
    station_id: 场站ID (1-10)
    pred_date: 预测日期的datetime对象
    
    返回:
    list: 包含96个默认预测值的列表
    """
    # 读取历史功率数据
    power_file = os.path.join(
        TRAIN_DATA_DIR, 'train_data', 'fact_data', f'{station_id}_normalization_train.csv'
    )
    
    try:
        if os.path.exists(power_file):
            print(f"读取历史功率数据: {power_file}")
            # 使用data_processor中的函数读取功率数据
            power_df = read_power_data(station_id)
            
            if power_df is None:
                raise Exception("无法读取功率数据")
            
            # 创建时间特征
            power_df['hour'] = power_df['datetime'].dt.hour
            power_df['minute'] = power_df['datetime'].dt.minute
            
            # 创建时间索引（0-95，对应一天中的96个15分钟间隔）
            time_indices = []
            for hour in range(24):
                for minute in [0, 15, 30, 45]:
                    time_indices.append((hour, minute))
            
            # 计算每个时间点的平均功率
            default_values = []
            for hour, minute in time_indices:
                avg_power = power_df[
                    (power_df['hour'] == hour) & 
                    (power_df['minute'] == minute)
                ]['power'].mean()
                
                if pd.isna(avg_power):
                    # 如果没有数据，使用基于时间的默认值
                    if station_id in WIND_STATION_IDS:
                        # 风电场站的默认值
                        time_idx = hour * 4 + minute // 15
                        avg_power = 0.02 + 0.005 * np.sin(2 * np.pi * (time_idx / 96))
                    else:
                        # 光伏场站的默认值，考虑日夜变化
                        if 6 <= hour < 18:  # 白天
                            avg_power = 0.03 + 0.02 * np.sin(np.pi * ((hour - 6) / 12))
                        else:  # 夜晚
                            avg_power = 0.005
                
                default_values.append(avg_power)
            
            print(f"成功生成基于历史数据的默认值，共 {len(default_values)} 个时间点")
            return default_values
        else:
            print(f"历史功率数据文件不存在: {power_file}")
            # 如果文件不存在，使用基于时间的默认值
            return generate_time_based_default_values(station_id, pred_date)
    except Exception as e:
        print(f"读取历史数据时出错: {e}，使用基于时间的默认值")
        return generate_time_based_default_values(station_id, pred_date)

def generate_time_based_default_values(station_id, pred_date):
    """
    基于时间生成默认预测值
    
    参数:
    station_id: 场站ID (1-10)
    pred_date: 预测日期的datetime对象
    
    返回:
    list: 包含96个默认预测值的列表
    """
    default_values = []
    
    for i in range(24 * 4):  # 24小时 * 每小时4个15分钟
        hour = (pred_date + timedelta(minutes=15 * i)).hour
        
        if station_id in WIND_STATION_IDS:
            # 风电场站的默认值
            default_values.append(0.02 + 0.005 * np.sin(2 * np.pi * (i / (24 * 4))))
        else:
            # 光伏场站的默认值，考虑日夜变化
            if 6 <= hour < 18:  # 白天
                default_values.append(0.03 + 0.02 * np.sin(np.pi * ((hour - 6) / 12)))
            else:  # 夜晚
                default_values.append(0.005)
    
    return default_values

def test_prediction_process():
    """
    测试预测流程，确保能够正确处理数据和进行预测
    """
    print("开始测试预测流程...")
    
    # 创建测试目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_dir = os.path.join(MODELS_DIR, f"test_{timestamp}")
    os.makedirs(test_dir, exist_ok=True)
    print(f"测试结果将保存到: {test_dir}")
    
    # 测试单个场站
    test_station_id = 1
    
    # 测试读取历史数据
    try:
        print(f"测试读取场站 {test_station_id} 的历史数据...")
        test_date = datetime.strptime('20250101', '%Y%m%d')
        default_values = generate_default_values_from_history(test_station_id, test_date)
        print(f"成功读取历史数据并生成默认值，共 {len(default_values)} 个时间点")
        print(f"默认值样例: {default_values[:5]}")
    except Exception as e:
        print(f"读取历史数据失败: {e}")
        return False
    
    # 测试加载训练数据
    try:
        print(f"测试加载场站 {test_station_id} 的训练数据...")
        # 使用1个月的训练数据进行测试
        X_train, X_test, y_train, y_test = prepare_data_for_station(
            test_station_id, 
            '20240101', 
            '20240131',  # 使用1月份的数据进行训练
            test_size=0.2
        )
        print(f"成功加载训练数据: X_train.shape={X_train.shape}, y_train.shape={y_train.shape}")
    except Exception as e:
        print(f"加载训练数据失败: {e}")
        return False
    
    # 测试模型训练
    try:
        print(f"测试训练场站 {test_station_id} 的模型...")
        model = train_model(X_train, y_train, 'xgboost', test_station_id)
        
        # 保存测试模型
        model_path = os.path.join(test_dir, f"station_{test_station_id}_xgboost_model.pkl")
        joblib.dump(model, model_path)
        print(f"测试模型已保存到 {model_path}")
        
        print(f"成功训练模型: {type(model).__name__}")
    except Exception as e:
        print(f"模型训练失败: {e}")
        return False
    
    # 测试单日预测
    try:
        print(f"测试预测场站 {test_station_id} 的单日功率...")
        predictions = predict_next_day(
            test_station_id, 
            model, 
            '20250101', 
            ['NWP_1']
        )
        print(f"成功预测单日功率: {predictions.shape if hasattr(predictions, 'shape') else '无预测结果'} 测试集形状: {X_test.shape}")
        if predictions is not None:
            # 显示更多的预测结果，展示24小时的预测
            print(f"预测结果样例 (24小时):\n{predictions.head(24)}")
            
            # 保存测试预测结果
            output_path = os.path.join(test_dir, f"output{test_station_id}.csv")
            predictions.to_csv(output_path)
            print(f"测试预测结果已保存到 {output_path}")
    except Exception as e:
        print(f"单日预测失败: {e}")
        return False
    
    print("预测流程测试成功!")
    return True

def main():
    """
    主函数
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='新能源功率预测模型')
    parser.add_argument('--test', action='store_true', help='测试模式')
    args = parser.parse_args()
    
    if args.test:
        # 测试模式
        test_prediction_process()
        return
    
    # 创建一个时间戳目录，用于保存所有结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(RESULTS_DIR, timestamp)
    os.makedirs(results_dir, exist_ok=True)
    print(f"所有预测结果将保存到: {results_dir}")
    
    # 训练日期范围
    train_start_date = '20240101'
    train_end_date = '20241230'  # 避开12月31日，因为没有数据
    
    # 预测日期范围
    prediction_start_date = '20250101'
    prediction_end_date = '20250228'
    
    # 气象源
    nwp_sources = ['NWP_1', 'NWP_2', 'NWP_3']
    
    # 模型类型
    model_type = 'xgboost'
    
    # 对每个场站进行训练和预测
    for station_id in range(1, 11):  # 场站ID从1到10
        try:
            print(f"处理场站 {station_id}...")
            
            # 准备数据
            X_train, X_test, y_train, y_test = prepare_data_for_station(
                station_id, 
                train_start_date, 
                train_end_date, 
                test_size=0.1
            )
            
            if X_train is None or y_train is None or X_train.empty or y_train.empty:
                print(f"无法获取场站 {station_id} 的训练数据，使用默认值进行预测")
                
                # 创建日期范围
                start_date_dt = datetime.strptime(prediction_start_date, '%Y%m%d')
                end_date_dt = datetime.strptime(prediction_end_date, '%Y%m%d')
                
                # 创建日期范围
                date_range = []
                current_date = start_date_dt
                while current_date <= end_date_dt:
                    date_range.append(current_date)
                    current_date += timedelta(days=1)
                
                # 创建默认预测结果
                all_predictions = []
                for pred_date in date_range:
                    default_df = create_default_prediction(station_id, pred_date, save_results=False)
                    
                    # 将默认预测结果添加到总预测结果中
                    for idx, row in default_df.iterrows():
                        all_predictions.append({
                            'datetime': idx,
                            'power': row['power']
                        })
                
                # 保存预测结果
                all_predictions_df = pd.DataFrame(all_predictions)
                if not all_predictions_df.empty:
                    all_predictions_df.set_index('datetime', inplace=True)
                    all_predictions_df = all_predictions_df.sort_index()
                    output_path = os.path.join(results_dir, f"output{station_id}.csv")
                    all_predictions_df.to_csv(output_path)
                    print(f"默认预测结果已保存到 {output_path}")
                
                continue
            
            # 训练模型
            print(f"训练场站 {station_id} 的模型...")
            model = train_model(X_train, y_train, model_type, station_id)
            
            # 保存模型
            model_path = os.path.join(results_dir, f"station_{station_id}_{model_type}_model.pkl")
            joblib.dump(model, model_path)
            print(f"模型已保存到 {model_path}")
            
            # 评估模型
            if X_test is not None and y_test is not None and not X_test.empty and not y_test.empty:
                print(f"评估场站 {station_id} 的模型...")
                metrics = evaluate_model(model, X_test, y_test)
                
                # 保存评估结果
                metrics_df = pd.DataFrame([metrics])
                metrics_df.to_csv(os.path.join(results_dir, f"station_{station_id}_metrics.csv"), index=False)
            
            # 预测多天功率
            print(f"预测场站 {station_id} 从 {prediction_start_date} 到 {prediction_end_date} 的功率...")
            predictions_df = predict_multiple_days(
                station_id, 
                model, 
                prediction_start_date, 
                prediction_end_date, 
                nwp_sources
            )
            
            # 保存预测结果
            output_path = os.path.join(results_dir, f"output{station_id}.csv")
            predictions_df.to_csv(output_path)
            print(f"预测结果已保存到 {output_path}")
            
        except Exception as e:
            print(f"处理场站 {station_id} 时出错: {e}")
            # 创建一个默认的预测结果，从2025/1/1到2025/2/28
            start_date_dt = datetime.strptime(prediction_start_date, '%Y%m%d')
            end_date_dt = datetime.strptime(prediction_end_date, '%Y%m%d')
            
            # 创建日期范围
            date_range = []
            current_date = start_date_dt
            while current_date <= end_date_dt:
                date_range.append(current_date)
                current_date += timedelta(days=1)
            
            # 创建默认预测结果
            all_predictions = []
            for pred_date in date_range:
                # 创建预测时间点
                for i in range(24 * 4):  # 24小时 * 每小时4个15分钟
                    pred_time = pred_date + timedelta(minutes=15 * i)
                    # 获取基于历史数据的默认值
                    hour = pred_time.hour
                    minute = pred_time.minute
                    time_idx = hour * 4 + minute // 15
                    default_values = generate_default_values_from_history(station_id, pred_date)
                    all_predictions.append({
                        'datetime': pred_time.strftime('%Y/%m/%d %H:%M'),
                        'power': default_values[time_idx]
                    })
            
            # 保存预测结果
            all_predictions_df = pd.DataFrame(all_predictions)
            all_predictions_df.set_index('datetime', inplace=True)
            output_path = os.path.join(results_dir, f"output{station_id}.csv")
            all_predictions_df.to_csv(output_path)
            print(f"默认预测结果已保存到 {output_path}")

if __name__ == "__main__":
    # 先运行测试
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_prediction_process()
    else:
        # 运行主程序
        main()
