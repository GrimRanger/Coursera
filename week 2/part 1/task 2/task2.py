import pandas
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import load_boston
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn import preprocessing

def find_average(X, y, p_val):
    kf = KFold(len(X), n_folds=5, shuffle=True,  random_state=42)

    neigh = KNeighborsRegressor(p = p_val, n_neighbors=5, metric ='minkowski', weights='distance')
    
    score = cross_val_score(neigh,X,y,cv=kf, scoring='mean_squared_error')
    average = reduce(lambda x, y: x + y, score) / len(score) 
   
    return average

def find_best(X, y):
    mas_result = []
    p_array =  np.linspace(1, 10, num=200)
    for i in p_array:
        result = find_average(X, y, i)
        mas_result.append(result)
        print str(i) + ":" + str(result)
    print "Max is :" + str(max(mas_result)) + " by index : " + str(np.argmax(mas_result))

if __name__ == "__main__":
    data = load_boston()
    X = data.data
    y = data.target
    X_scaled = preprocessing.scale(X)
    find_best(X_scaled, y)

   
    
