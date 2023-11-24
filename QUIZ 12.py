import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, KFold
from sklearn.tree import DecisionTreeClassifier
from pandas.plotting import scatter_matrix

# 데이터 파일 경로
filename = "./data/09_irisdata.csv"

# 컬럼명 정의
column_names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']

# 데이터 불러오기
data = pd.read_csv(filename, names=column_names)

# 나머지 코드는 이전에 제공한 코드와 동일합니다.
# (행렬 크기, describe(), 클래스 종류, scatter_matrix, 분할, 의사결정 나무, K-fold 등)

# 데이터 셋의 행렬 크기(shape)
print("데이터 셋의 행렬 크기:", data.shape)

# 데이터 셋의 요약(describe())
print("데이터 셋의 요약:\n", data.describe())

# 데이터 셋의 클래스 종류(groupby('class').size())
print("데이터 셋의 클래스 종류:\n", data.groupby('class').size())

# scatter_matrix 그래프 저장
scatter_matrix(data, alpha=0.8, figsize=(10, 8), diagonal='kde')
plt.savefig('scatter_matrix.png')

# 독립 변수 X와 종속 변수 Y로 분할
X = data.iloc[:, 0:4]
Y = data.iloc[:, 4]

# 의사결정 나무 모델 생성
model = DecisionTreeClassifier()

# K-fold(10개의 폴드 지정), cross validation(평가 지표 accuracy)
kfold = KFold(n_splits=10, random_state=42, shuffle=True)
results = cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')

# K-fold의 평균 정확도 출력
print("K-fold의 평균 정확도:", np.mean(results))
