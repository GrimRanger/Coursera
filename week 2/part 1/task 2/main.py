import pandas
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn import preprocessing

def find_average(X, y, neighbors):
    kf = KFold(len(X), n_folds=5, shuffle=True,  random_state=42)

    neigh = KNeighborsClassifier(n_neighbors=neighbors)
    
    score = cross_val_score(neigh,X,y,cv=kf, scoring='accuracy')
    average = reduce(lambda x, y: x + y, score) / len(score) 
   
    return average

def find_average_and_neighbors(X, y):
    mas_result = []
    for i in range(1, 51):
        result = find_average(X, y, i)
        mas_result.append(result)
        print str(i) + ":" + str(result)
    print "Max is :" + str(max(mas_result)) + " by index : " + str(np.argmax(mas_result))

if __name__ == "__main__":
    data = pandas.read_csv('wine.data.txt')
    y = data[data.columns[0]]
    X = data.drop(data.columns[0], axis=1)
    find_average_and_neighbors(X, y)
    X_scaled = preprocessing.scale(X)

    find_average_and_neighbors(X_scaled, y)
   
    
