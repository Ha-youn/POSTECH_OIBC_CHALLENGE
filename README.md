
---

# POSTECH\_OIBC\_CHALLENGE

이 프로젝트는 **2023 POSTECH OIBC CHALLENGE**에 참가하여 진행한 태양광 발전량 예측 모델 개발에 관한 것입니다.

---

## 개요

* **대회명**: 제5회 POSTECH OIBC CHALLENGE
* **주제**: 다종 태양광 발전 예측 모형의 앙상블 활용 방안 개발
* **기간**: 2023년 10월 \~ 11월
* **참가자**: Ha-youn, Jujack0x0
* **주최**: POSTECH 오픈이노베이션 빅데이터센터(OIBC), H에너지
* **수상**: 장려상

---

## 사용 기술

* Python 3.10
* Pandas, NumPy (데이터 처리 및 수치 계산)
* TensorFlow (Keras API 기반 LSTM/GRU 모델 구현)
* Scikit-learn (MinMaxScaler, StandardScaler 등 정규화 및 전처리)
* Requests, JSON, Datetime (기상 API 연동 및 시계열 처리)

---

## 프로젝트 구조

```
POSTECH_OIBC_CHALLENGE/
├── data1/                     # 기상 및 발전량 관련 데이터
├── API_제출코드.py            # API 데이터를 이용한 예측 모델 제출 코드
├── GENS_제출코드.py           # GENS 데이터 기반 예측용 코드
├── weather_GRU_제출코드.py    # GRU 기반 기상 모델 예측 코드
├── 메인모델_제출코드.py       # 통합된 예측 모델 실행 파일
└── README.md                  # 프로젝트 설명 문서
```

---

## 주요 기능

1. **데이터 수집 및 전처리**

   * 기상 API와 발전소 데이터를 정규화 및 구조화

2. **예측 모델 학습**

   * GRU 및 LSTM 기반 시계열 예측 모델 구성 및 학습

3. **앙상블 및 예측**

   * 다수의 모델 예측값을 조합하여 성능 향상

4. **성능 평가 및 결과 출력**

   * 실제 발전량과의 비교를 통해 예측 정확도 평가

---

## 실행 방법

1. 필수 라이브러리 설치:

   ```bash
   pip install -r requirements.txt
   ```

2. 예측 코드 실행:

   ```bash
   python 메인모델_제출코드.py
   ```

   또는 각 개별 코드 실행:

   ```bash
   python API_제출코드.py
   python weather_GRU_제출코드.py
   python GENS_제출코드.py
   ```

---

## 참고

* POSTECH OIBC 공식 홈페이지: [https://oibc.postech.ac.kr/](https://oibc.postech.ac.kr/)
* 기상청 Open API, GENS 데이터 활용

---
