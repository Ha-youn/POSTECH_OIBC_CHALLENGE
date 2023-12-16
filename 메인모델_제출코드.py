import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler

# 파일에 있는 데이터 불러오기
weather_actual_data = pd.read_csv("C:/Users/PC/Desktop/data1/weather_actual.csv")
gens_data = pd.read_csv("C:/Users/PC/Desktop/data1/gens.csv")
weather_actual_test_data = pd.read_csv("C:/Users/PC/Desktop/data1/predictions_we.csv")
gens_test_data = pd.read_csv("C:/Users/PC/Desktop/data1/predictions_gn.csv")

# 데이터 프레임으로 변환
weather_actual_test_data = pd.DataFrame(weather_actual_test_data)
gens_test_data = pd.DataFrame(gens_test_data)

# 데이터 가중치 적용 gens 데이터에 0.8, weather 데이터에 0.2
weather_actual_test_data = weather_actual_test_data.drop(columns=['time'])
gens_test_data_weighted = gens_test_data * 0.8
weather_actual_test_data_weighted = weather_actual_test_data * 0.2
merged_df = pd.concat([weather_actual_test_data_weighted, gens_test_data_weighted], axis=1)

# 중복되는 'time' 열을 제거하고 데이터에도 가중치를 적용
weather_actual_data = weather_actual_data.drop(columns=['time'])
gens_data = gens_data.drop(columns=['time'])
gens_data_weighted = gens_data * 0.8
weather_actual_data_weighted = weather_actual_data * 0.2
merge_train = pd.concat([weather_actual_data_weighted, gens_data_weighted], axis=1)

# 데이터를 시계열 시퀀스로 변환
sequence_length = 3
sequences = []

for i in range(len(merge_train) - sequence_length):
    sequence = merge_train.iloc[i:i + sequence_length, :].values
    sequences.append(sequence)

# 시계열 데이터를 NumPy 배열로 변환
sequences = np.array(sequences)

# 데이터와 레이블 분리
X = sequences[:, :, :]
y = merge_train.iloc[sequence_length:, -1].values  # 마지막 열이 발전량

# 데이터을 float32로 유형 변환
X = X.astype('float32')
y = y.astype('float32')

# 새로운 스케일러 생성
scaler = StandardScaler()

# 스케일러 적용
X_scaled = scaler.fit_transform(X.reshape(X.shape[0], -1)).reshape(X.shape)
y_scaled = scaler.fit_transform(y.reshape(-1, 1)).reshape(y.shape)
merged_df_array_scaled = scaler.fit_transform(merged_df.values.reshape(-1, 1)).reshape(merged_df.shape)

# 모델 형태 지정
model = Sequential()
model.add(LSTM(50, input_shape=(3, 14)))
model.add(Dense(1))

# 모델 컴파일 옵티마이저로 adam 사용하고 loss는 MSE 이용
model.compile(optimizer='adam', loss='mean_squared_error')

split_index = int(0.8 * len(X_scaled))
X_train, X_val = X_scaled[:split_index], X_scaled[split_index:]
y_train, y_val = y_scaled[:split_index], y_scaled[split_index:]

# 데이터를 이용하여 모델 학습
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val))

# 테스트할 데이터를 합쳐서 test_row로 
test_row = merged_df.iloc[:3, :].values.reshape(1, -1)

# LSTM에 넣기 위해서 (1,3,14)로 크기 맞춰주기
test_row = test_row.reshape(1, 3, 14)

# test_row을 스케일 하기
test_row_scaled = scaler.transform(test_row.reshape(-1, 1)).reshape(test_row.shape)


# 24시간 동안의 값 예측
for i in range(24):
    prediction = model.predict(test_row_scaled)

    # 예측할 배열의 첫번째값으로 시작
    predicted_value = prediction[0, 0]  

    # 스케일 역변환
    predicted_value = scaler.inverse_transform(predicted_value.reshape(-1, 1))

    # 예측할 배열의 다음 배열로 넘어감
    test_row_scaled = np.roll(test_row_scaled, shift=-1)
    test_row_scaled[-1, 0] = scaler.transform(predicted_value)[0, 0]
    
    #태양광 발전이 없는 시간대는 0으로 출력
    if 6 <= i <= 18:
        print(f"Prediction for time step {i + 1}: {predicted_value}")
    else:
        print(f"Prediction for time step {i + 1}: 0")
