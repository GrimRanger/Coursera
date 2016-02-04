import pandas
import numpy as np
from sklearn.svm import SVC


def learn(X,y):
    clf = SVC(C = 100000, random_state=241, kernel='linear')
    clf.fit(X, y)
   
    return clf

def test(X, y, clf):
    y_actual = clf.predict(X)
    result = accuracy_score(y, y_actual)
    
    return result

def load(file_name):
    data = pandas.read_csv(file_name)
    y = data[data.columns[0]]
    X = data.drop(data.columns[0], axis=1)

    return X, y

def test_SVC():
    X, y = load('svm-data.csv')
    print X[0:1]
    clf = learn(X, y)
    print clf.support_
    
if __name__ == "__main__":
   test_SVC()
  
