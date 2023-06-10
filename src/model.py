import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris


class IrisClassifier:
    def __init__(self):
        self.model = KNeighborsClassifier(n_neighbors=3)
    
    
    def train(self, X, y):
        self.model.fit(X, y)
    

    def predict(self, X):
        return self.model.predict(X)
    

    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        accuracy = accuracy_score(y_true, y_pred)
        return accuracy
