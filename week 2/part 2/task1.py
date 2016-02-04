import pandas
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def learn(X,y):
    clf = Perceptron(random_state=241)
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

def test_perceptron():
    X_test, y_test = load('perceptron-test.csv')
    X_train, y_train = load('perceptron-train.csv')
    clf = learn(X_train, y_train)
    result1 = test(X_test, y_test, clf)
    print "before :" + str(result1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    clf = learn(X_train_scaled, y_train)
    result2 = test(X_test_scaled, y_test, clf)
    print "after :" + str(result2)
    print "diff :" + str(result2 - result1)
 
if __name__ == "__main__":
   test_perceptron()
  
