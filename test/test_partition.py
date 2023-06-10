#70%의 훈련 데이터와 30%의 테스트 데이터로 분할하는 예시

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


iris = load_iris()


X = iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


print("훈련 데이터 shape:", X_train.shape)
print("테스트 데이터 shape:", X_test.shape)
