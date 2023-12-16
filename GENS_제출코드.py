import pandas as pd
import numpy as np
import requests

# API주소 및 키 지정
_API_URL = "https://research-api.solarkim.com"
_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJiM2dIQWtRSjlDQzRmWVJ3SnJ1VVRZIiwiaWF0IjoxNjk4NTg3MDYyLCJleHAiOjE3MDAyMzMyMDAsInR5cGUiOiJhcGlfa2V5In0.4NT3gHQSZ4lVOTZEXgvD4Mc5VVKg_YpBW42EaMcbmC0" 
_AUTH_PARAM = {"headers": {"Authorization": f"Bearer {_API_KEY}"}}

# API의 url, 내용 조회
def _get(url: str):
    response = requests.get(url, **_AUTH_PARAM)
    return response.json()

def _get_gen_forecasts():   
    date = "2023-11-17"     # 날짜 지정하여 이용
    bid_round_10 = 1
    bid_round_17 = 2

    # API를 통해 데이터 가져오기
    gen_fcst_10 = _get(f"{_API_URL}/cmpt-2023/gen-forecasts/{date}/{bid_round_10}")
    gen_fcst_17 = _get(f"{_API_URL}/cmpt-2023/gen-forecasts/{date}/{bid_round_17}")
    
    # 데이터프레임 생성
    data_10 = pd.DataFrame(gen_fcst_10)
    data_17 = pd.DataFrame(gen_fcst_17)
    
    print(data_10)
    # 필요한 데이터 반환
    return data_10, data_17

# 데이터 행에서 가장 작은값과 가장 큰 값을 제외한 나머지 세 값의 평균을 반환
def predictions_gn(row):
     # 'None' 값을 NaN으로 대체하고 수치형으로 변환
    num_values = pd.to_numeric(row.replace('0', np.nan), errors='coerce')

    # '시간' 열을 제외하고 최대값과 최소값을 찾음
    max_value = num_values[num_values.index != 'time'].max()
    min_value = num_values[num_values.index != 'time'].min()

    # 최대값과 최소값을 제외한 값들로 이루어진 Series 생성
    res = num_values[((num_values != max_value) & (num_values.index != 'time')) & ((num_values != min_value) & (num_values.index != 'time'))]
   
    # NaN 값은 0으로 대체
    res = res.fillna('0')
    
    # 남은 값들의 평균 계산
    return res.mean()

# 데이터 가져오기
data_10 = _get_gen_forecasts() 
data_17 = _get_gen_forecasts() 

# 함수 적용하여 각 행의 결과 얻기
res_10 = data_10.apply(predictions_gn, axis=1)
res_17 = data_17.apply(predictions_gn, axis=1)

# 결과를 CSV 파일로 저장
res_10.to_csv("predictions_gn_10.csv", header=["amount"], index=False, encoding='utf-8-sig')
res_17.to_csv("predictions_gn_17.csv", header=["amount"], index=False, encoding='utf-8-sig')