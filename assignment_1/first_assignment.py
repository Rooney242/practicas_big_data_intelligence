import pandas as pd
from sklearn import preprocessing, model_selection
from sklearn import tree, neighbors
from sklearn import metrics


data = pd.read_csv("kaggleCompetition.csv")
data = data.values

x = data[0:1460, :-1]
y = data[0:1460, -1] 
x_comp = data[1460:,:-1] 
y_comp = data[1460:,-1]

# standardize input attributes
scaler = preprocessing.StandardScaler().fit(x) 
x = scaler.transform(x)
x_comp = scaler.transform(x_comp)

#print(x_comp)
#print(y_comp)

# split in train/test sets
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.75)

# decision tree for regression
clf = tree.DecisionTreeRegressor()
clf = clf.fit(x_train, y_train)

y_train_pred = clf.predict(x_train)
y_test_pred = clf.predict(x_test)

print('decision tree evaluation')
print('\tmse on train set:', metrics.mean_squared_error(y_train, y_train_pred))
print('\tmse on test set:', metrics.mean_squared_error(y_test, y_test_pred))

# regression based on knn
clf = neighbors.KNeighborsRegressor()
clf = clf.fit(x_train, y_train)

y_train_pred = clf.predict(x_train)
y_test_pred = clf.predict(x_test)

print('knn evaluation')
print('\tmse on train set:', metrics.mean_squared_error(y_train, y_train_pred))
print('\tmse on test set:', metrics.mean_squared_error(y_test, y_test_pred))
