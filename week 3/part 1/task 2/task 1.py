import pandas
import math
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.cross_validation import cross_val_score


def find_score(X,y, C_value, length):
   
    kf = KFold(length, n_folds=5, shuffle=True,  random_state=42)   
    svc = SVC(C = C_value, random_state=241, kernel='linear')
    score = cross_val_score(svc, X, y, cv=kf, scoring='accuracy')
    average = reduce(lambda x, y: x + y, score) / len(score) 
   
    return average

def find_best_param_c_by_cross_validation(X, y, length):
    params_c = [math.pow(10, i) for i in range(-5, 6)]
    results = []
    #for c in params_c:
        #score = find_score(X, y, c, length)
        #results.append(score)
    #print results
    index_of_max = 5
    #index_of_max = np.argmax(results)
    #print index_of_max
    return params_c[index_of_max]
    
def load():
    newsgroups = fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])   
    y = newsgroups.target
    X = newsgroups.data
    length = len(X)
   
    tv = TfidfVectorizer()
    X_transformed = tv.fit_transform(X)
   
    return X_transformed, y, length

def learn(X,y, c_value):
    clf = SVC(C = c_value, random_state=241, kernel='linear')
    clf.fit(X, y)
   
    return clf

def test_SVC():
    X, y, length = load()
    c = find_best_param_c_by_cross_validation(X, y, length)
    clf = learn(X,y,c)
    result = clf.coef_
    print np.argmax(result)
    print result
    
if __name__ == "__main__":
   test_SVC()
  
