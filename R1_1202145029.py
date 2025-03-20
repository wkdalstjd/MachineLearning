import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from scipy import stats

# CSV 파일 불러오기
file_path = r"C:/Users/wkdal/OneDrive/바탕 화면/exam112.csv"  # 파일 경로 수정
df = pd.read_csv(file_path)

# 결측치 개수 확인
print("결측치 개수 (처리 전):\n", df.isnull().sum())

# 수치형 데이터 → 중앙값(median)으로 대체
num_cols = df.select_dtypes(include=["number"]).columns  # 숫자형 컬럼 선택
num_imputer = SimpleImputer(strategy="median")  # 중앙값(median) 대체
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# (2) 범주형 데이터 → 최빈값(mode)으로 대체
cat_cols = df.select_dtypes(include=["object"]).columns  # 문자형 컬럼 선택
cat_imputer = SimpleImputer(strategy="most_frequent")  # 최빈값(mode) 대체
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

# (3) KNN Imputer를 활용한 결측치 보완
knn_imputer = KNNImputer(n_neighbors=3)  # KNN 이웃 3개 참고
df[num_cols] = knn_imputer.fit_transform(df[num_cols])  # 숫자형 데이터 보완

# 결측치 처리 후 확인
print("결측치 개수 (처리 후):\n", df.isnull().sum())

# 변경된 데이터 저장
df.to_csv("exam112_cleaned.csv", index=False)

#이상값 처리
def remove_outliers_zscore(df, column, threshold=3):
    z_scores = np.abs(stats.zscore(df[column]))  # Z-Score 계산
    return df[z_scores < threshold]  # 임계값(threshold)보다 큰 값 제거

# 모든 숫자형 컬럼에 적용
for col in num_cols:
    df = remove_outliers_zscore(df, col)

#이상치 처리 후 데이터 크기 확인
print("이상치 처리 후 데이터 크기:", df.shape)

#변경된 데이터 저장
df.to_csv("exam112_cleaned_outliers.csv", index=False)