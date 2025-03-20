import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# 데이터 로드 및 전처리
file_path = r"C:/파이썬/exam112_cleaned_outliers.csv"
df = pd.read_csv(file_path)

df = df[["math", "eng", "science", "pass"]]
df.dropna(inplace=True)  # 결측치 제거

# 학습 데이터 분리
X = df[["math", "eng", "science"]]  # 입력 변수 (수학, 영어, 과학 점수)
y = df["pass"]  # 출력 변수 (합격 여부)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# KNN 모델 학습
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# 점수 입력 → 합격 여부 예측 함수
def predict_pass_fail(math_score, eng_score, science_score):
    prediction = model.predict([[math_score, eng_score, science_score]])[0]
    return prediction

# 점수 입력 시 합격 여부 예측
math_score = int(input("수학 점수를 입력하세요: "))
eng_score = int(input("영어 점수를 입력하세요: "))
science_score = int(input("과학 점수를 입력하세요: "))
predicted_result = predict_pass_fail(math_score, eng_score, science_score)
print(f"예측된 합격 여부: {predicted_result}")
