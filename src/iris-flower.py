# ver 0.0.1

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris


class IrisClassifier:
    def __init__(self):
        # K-Nearest Neighbors 분류기 객체 생성.
        self.model = KNeighborsClassifier(n_neighbors=3)


    def train(self, X, y):
        # 분류기 훈련 데이터로 학습.
        self.model.fit(X, y)


    def predict(self, X):
        # 분류기 사용, 입력 데이터 클래스를 예측.
        return self.model.predict(X)


    def evaluate(self, X, y_true):
        # 테스트 데이터 예측 수행 및 정확도 계산.
        y_pred = self.predict(X)
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy


# IRIS 데이터셋 로드
iris = load_iris()


# 특성 데이터와 타겟 데이터 설정
X = iris.data
y = iris.target


# 훈련 데이터와 테스트 데이터로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


# 분류기 객체 생성 및 훈련
classifier = IrisClassifier()
classifier.train(X_train, y_train)


# 테스트 데이터로 예측 및 평가
accuracy = classifier.evaluate(X_test, y_test)
print("정확도:", accuracy)




