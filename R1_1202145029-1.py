import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 데이터 로드 및 전처리
file_path = r"C:/파이썬/exam112_cleaned_outliers.csv"
df = pd.read_csv(file_path)

df = df[["math", "science"]]
df.dropna(inplace=True)  # 결측치 제거

# 학습 데이터 분리
X = df[["math"]]  # 입력 변수 (수학 점수)
y = df["science"]  # 출력 변수 (과학 점수)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN 모델 학습
model = KNeighborsRegressor(n_neighbors=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# 과학 점수 예측 함수
def predict_science(math_score):
    predicted_science = model.predict([[math_score]])[0]
    return round(predicted_science, 2)

# 수학 점수 입력 받고 과학 점수 예측
math_score = int(input("점수를 입력하세요: "))
predicted_science = predict_science(math_score)
print(f"예상 과학 점수: {predicted_science}")

# 데이터 시각화
plt.scatter(X_test, y_test, color='blue', label="실제 값")
plt.scatter(X_test, y_pred, color='red', label="예측 값")
plt.xlabel("수학 점수")
plt.ylabel("과학 점수")
plt.title("과학 점수 예측하기")
plt.legend()
plt.show()
