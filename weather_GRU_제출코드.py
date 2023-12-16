import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import requests
from datetime import datetime, timedelta

# API 정보 입력
_API_URL = "https://research-api.solarkim.com"
_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiM2dIQWtRSjlDQzRmWVJ3SnJ1VVRZIiwiaWF0IjoxNjk4NTg3MDYyLCJleHAiOjE3MDAyMzMyMDAsInR5cGUiOiJhcGlfa2V5In0.4NT3gHQSZ4lVOTZEXgvD4Mc5VVKg_YpBW42EaMcbmC0" 
_AUTH_PARAM = {"headers": {"Authorization": f"Bearer {_API_KEY}"}}

# API 상태 확인
response = requests.get(_API_URL)

# API 함수 선언
def _get(url: str):
    response = requests.get(url, **_AUTH_PARAM)
    return response.json()

# API로 10시,17시 예측 날씨 불러오기
def _get_weathers_forecasts(variable, date):
    date = "2023-11-23"
    bid_round_10 = 1
    bid_round_17 = 2
    weather_fcst_10 = _get(f"{_API_URL}/cmpt-2023/weathers-forecasts/{date}/{bid_round_10}")
    weather_fcst_17 = _get(f"{_API_URL}/cmpt-2023/weathers-forecasts/{date}/{bid_round_17}")

    data_10 = []
    for item in weather_fcst_10:
        if variable in item:
            data_10.append(item[variable])
    
    data_17 = []
    for item in weather_fcst_17:
        if variable in item:
            data_17.append(item[variable])
    
    return data_10, data_17

# 훈련 모델로 함수 relu와 mse 사용
def create_model(input_shape, output_dim):
    model = Sequential()
    model.add(GRU(64, input_shape=input_shape, activation='relu'))
    model.add(Dense(output_dim))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
    return model

# 훈련 평가 모델
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train, epochs=50, batch_size=3)
    loss = model.evaluate(X_test, y_test)
    return loss

# 파일로 주어진 데이터 불러오기
weather_actual_data = pd.read_csv('./data1/weather_actual.csv')
weather_forecast_data = pd.read_csv('./data1/weather_forecast.csv')

# 날짜
current_date = datetime.strptime('2023-11-17', '%Y-%m-%d')

# 다음 날짜 설정 (24시간 후)
next_date = current_date + timedelta(days=1)
next_date_str = next_date.strftime('%Y-%m-%d')

# 훈련시킬 항목 리스트
variables_to_predict = ['cloud','temp','humidity','ground_press','wind_speed','wind_dir','rain','snow','dew_point','vis','uv_idx','azimuth','elevation']  # Add more if needed

# 훈련 데이터를 데이터 프레임으로 제작
predicted_df = pd.DataFrame(columns=['time', 'cloud', 'temp', 'humidity','ground_press','wind_speed','wind_dir','rain','snow','dew_point','vis','uv_idx','azimuth','elevation'])

for variable in variables_to_predict:
    
    # 훈련시킬 데이터 함수에서 가져오기
    data_10, data_17 = _get_weathers_forecasts(variable, next_date_str)

    # 데이터 형식을 구체화 하기
    actual_data = weather_actual_data[variable].values
    forecast_data = weather_forecast_data[variable].values
    scaler = MinMaxScaler()
    actual_data = scaler.fit_transform(actual_data.reshape(-1, 1))
    forecast_data = scaler.transform(forecast_data.reshape(-1, 1))
    seq_length = 24
    X, y = [], []

    # 데이터의 크기 알아내기
    for i in range(len(actual_data) - seq_length):
        X.append(actual_data[i:i+seq_length])
        y.append(forecast_data[i+seq_length])
    X = np.array(X)
    y = np.array(y)

    # 훈련데이터와 테스트 데이터 분리
    split_ratio = 0.8
    split_index = int(len(X) * split_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # 데이터 훈련
    model = create_model(input_shape=(seq_length, 1), output_dim=1)
    train_loss = train_and_evaluate_model(model, X_train, y_train, X_test, y_test)
    
    future_input_data = np.array(data_10)
    future_input_data = scaler.transform(future_input_data.reshape(-1, 1))

    # 예측된 값을 저장할 리스트 초기화
    predicted_values = []
    future_input_data = future_input_data.reshape(1, seq_length, 1)

    # 미래 24시간 예측
    for i in range(24):
        predicted_value = model.predict(future_input_data)
        predicted_value = scaler.inverse_transform(predicted_value)
        predicted_values.append(predicted_value[0][0])
        future_input_data = np.roll(future_input_data, shift=-1)
    
    # 예측값 DataFrame에 추가
    predicted_df[variable] = predicted_values[:24]    
    
    # 출력
    for hour, value in enumerate(predicted_values):
        print(f'{variable} - {hour+1}시간 후 예측값: {value:.6f}')  

# 시간(time) 정보 추가
predicted_df['time'] = range(1, 25)

# CSV 파일로 예측값 저장
predicted_df.to_csv(f"predictions_we.csv", index=False, encoding='utf-8-sig')
print(f'예측값을 predictions_we.csv에 저장했습니다.')