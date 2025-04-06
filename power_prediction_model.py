import os
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
from data_processor import process_station_data, align_power_and_weather, create_features, WIND_STATION_IDS

# 定义常量
MODELS_DIR = "models"
RESULTS_DIR = "results"

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
    y_train: 训练目标
    model_type: 模型类型，可选值: 'xgboost', 'lightgbm', 'random_forest', 'gradient_boosting', 'linear', 'svr'
    station_id: 场站ID，用于保存模型
    
    返回:
    model: 训练好的模型
    """
    print(f"使用 {model_type} 模型训练...")
    
    if model_type == 'xgboost':
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_child_weight=1,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='reg:squarederror',
            random_state=42
        )
    elif model_type == 'lightgbm':
        model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='regression',
            random_state=42
        )
    elif model_type == 'random_forest':
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
    elif model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'svr':
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR(kernel='rbf', C=1.0, epsilon=0.1))
        ])
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 保存模型
    if station_id is not None:
        model_path = os.path.join(MODELS_DIR, f"station_{station_id}_{model_type}_model.pkl")
        joblib.dump(model, model_path)
        print(f"模型已保存到 {model_path}")
    
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
    plt.savefig(os.path.join(RESULTS_DIR, 'prediction_comparison.png'))
    plt.close()
    
    # 可视化预测误差
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.values - y_pred)
    plt.title('预测误差')
    plt.xlabel('样本索引')
    plt.ylabel('误差')
    plt.savefig(os.path.join(RESULTS_DIR, 'prediction_error.png'))
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
    
    # 获取前一天的气象预报数据
    previous_date = (prediction_date_dt - timedelta(days=1)).strftime('%Y%m%d')
    
    # 处理气象数据
    _, weather_df = process_station_data(station_id, previous_date, previous_date, nwp_sources=nwp_sources)
    
    if weather_df is None or weather_df.empty:
        print(f"无法获取场站 {station_id} 的气象预报数据，使用默认值生成预测结果")
        # 创建一个默认的预测结果
        return create_default_prediction(station_id, prediction_date_dt)
    
    # 创建预测时间点
    prediction_times = []
    start_time = prediction_date_dt
    for i in range(24 * 4):  # 24小时 * 每小时4个15分钟
        prediction_times.append(start_time + timedelta(minutes=15 * i))
    
    # 创建一个空的DataFrame来存储预测结果
    predictions = []
    
    # 记录上一个有效的预测值，用于填充缺失值
    last_valid_prediction = 0.0
    
    # 对于每个预测时间点
    for pred_time in prediction_times:
        # 找到对应的气象预报数据
        forecast_date = previous_date
        forecast_hour = pred_time.hour
        
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
            features['day'] = pred_time.day
            features['month'] = pred_time.month
            features['dayofweek'] = pred_time.weekday()
            features['is_weekend'] = int(pred_time.weekday() in [5, 6])
            features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
            features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
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
            
            # 添加风速特征 (对于风电场站)
            if station_id in WIND_STATION_IDS:
                for nwp_source in nwp_sources:
                    if f'{nwp_source}_u100' in features and f'{nwp_source}_v100' in features:
                        features[f'{nwp_source}_wind_speed'] = np.sqrt(
                            features[f'{nwp_source}_u100']**2 + 
                            features[f'{nwp_source}_v100']**2
                        )
                        features[f'{nwp_source}_wind_direction'] = np.arctan2(
                            features[f'{nwp_source}_v100'], 
                            features[f'{nwp_source}_u100']
                        )
            
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
            last_valid_prediction = power_pred  # 更新上一个有效预测值
            
            # 添加到预测结果
            predictions.append({
                'datetime': pred_time,
                'power': power_pred  # 不再乘以100
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
    
    # 保存预测结果
    output_path = os.path.join(RESULTS_DIR, f"output{station_id}.csv")
    predictions_df.to_csv(output_path)
    print(f"预测结果已保存到 {output_path}")
    
    return predictions_df

def create_default_prediction(station_id, prediction_date_dt):
    """
    当无法获取气象数据时，创建默认的预测结果
    
    参数:
    station_id: 场站ID (1-10)
    prediction_date_dt: 预测日期的datetime对象
    
    返回:
    pandas.DataFrame: 包含默认预测结果的DataFrame
    """
    print(f"为场站 {station_id} 创建默认预测结果")
    
    # 创建预测时间点
    prediction_times = []
    for i in range(24 * 4):  # 24小时 * 每小时4个15分钟
        prediction_times.append(prediction_date_dt + timedelta(minutes=15 * i))
    
    # 根据场站类型设置默认值
    if station_id in WIND_STATION_IDS:
        # 风电场站的默认值
        default_values = [0.02 + 0.005 * np.sin(2 * np.pi * (i / (24 * 4))) for i in range(24 * 4)]
    else:
        # 光伏场站的默认值，考虑日夜变化
        default_values = []
        for i in range(24 * 4):
            hour = (prediction_date_dt + timedelta(minutes=15 * i)).hour
            if 6 <= hour < 18:  # 白天
                default_values.append(0.03 + 0.02 * np.sin(np.pi * ((hour - 6) / 12)))
            else:  # 夜晚
                default_values.append(0.005)
    
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
    
    # 保存预测结果
    output_path = os.path.join(RESULTS_DIR, f"output{station_id}.csv")
    predictions_df.to_csv(output_path)
    print(f"默认预测结果已保存到 {output_path}")
    
    return predictions_df

def main():
    """
    主函数
    """
    # 创建必要的目录
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 设置参数 - 完整版本
    station_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 所有场站
    start_date = '20240101'  # 训练数据开始日期
    end_date = '20241230'    # 训练数据结束日期（扩展到一年）
    model_type = 'xgboost'   # 模型类型
    prediction_date = '20241231'  # 预测日期
    nwp_sources = ['NWP_1', 'NWP_2', 'NWP_3']  # 所有气象源
    
    # 处理每个场站
    for station_id in station_ids:
        print(f"\n{'='*50}")
        print(f"处理场站 {station_id}")
        print(f"{'='*50}")
        
        try:
            # 准备数据
            print(f"准备场站 {station_id} 的数据...")
            X_train, X_test, y_train, y_test = prepare_data_for_station(
                station_id, start_date, end_date, test_size=0.2
            )
            
            if X_train is None or X_train.empty or y_train is None or y_train.empty:
                print(f"无法准备场站 {station_id} 的训练数据，使用默认模型")
                # 创建一个默认的预测结果
                create_default_prediction(station_id, datetime.strptime(prediction_date, '%Y%m%d'))
                continue
            
            # 训练模型
            print(f"训练场站 {station_id} 的模型...")
            model = train_model(X_train, y_train, model_type, station_id)
            
            # 评估模型
            if X_test is not None and y_test is not None and not X_test.empty and not y_test.empty:
                print(f"评估场站 {station_id} 的模型...")
                metrics = evaluate_model(model, X_test, y_test)
                
                # 保存评估结果
                metrics_df = pd.DataFrame([metrics])
                metrics_df.to_csv(os.path.join(RESULTS_DIR, f"station_{station_id}_metrics.csv"), index=False)
            
            # 预测次日功率
            print(f"预测场站 {station_id} 的次日功率...")
            predictions = predict_next_day(station_id, model, prediction_date, nwp_sources)
            
            if predictions is not None:
                print(f"场站 {station_id} 的预测完成!")
                print(predictions.head())
        except Exception as e:
            print(f"处理场站 {station_id} 时出错: {e}")
            # 创建一个默认的预测结果
            create_default_prediction(station_id, datetime.strptime(prediction_date, '%Y%m%d'))
            continue

if __name__ == "__main__":
    main()
