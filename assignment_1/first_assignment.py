import pandas as pd
import numpy as np
from sklearn import preprocessing, model_selection
from sklearn import tree, neighbors
from sklearn import metrics
from scipy.stats import uniform, randint as sp_randint


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
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.75, random_state=0)

# FIXED HYPERPARAM
# decision tree for regression
clf = tree.DecisionTreeRegressor(random_state=0)
clf = clf.fit(x_train, y_train)

y_train_pred = clf.predict(x_train)
y_test_pred = clf.predict(x_test)

print('decision tree evaluation')
print('\tmse on train set:', metrics.mean_squared_error(y_train, y_train_pred, squared=False))
print('\tmse on test set:', metrics.mean_squared_error(y_test, y_test_pred, squared=False))

# regression based on knn
clf = neighbors.KNeighborsRegressor()
clf = clf.fit(x_train, y_train)

y_train_pred = clf.predict(x_train)
y_test_pred = clf.predict(x_test)

print('knn evaluation')
print('\tmse on train set:', metrics.mean_squared_error(y_train, y_train_pred, squared=False))
print('\tmse on test set:', metrics.mean_squared_error(y_test, y_test_pred, squared=False))
print()

# HYPERPARAM TUNING THROUGH RANDOM SEARCH
print('hyperparameter tuning through random search')
cv_grid = model_selection.KFold(n_splits=2, shuffle=True, random_state=0) # hyperparams evaluated by 2-fold CV

# Decision tree
np.random.seed(0)
param_grid = {'min_samples_split': uniform.rvs(size=15),
			  'max_depth': sp_randint(2,16)}
budget = 20

np.random.seed(0)
clf_tune_tree = model_selection.RandomizedSearchCV(tree.DecisionTreeRegressor(random_state=0), 
                         param_grid,
                         cv=cv_grid, 
                         n_jobs=1, verbose=1,
                         n_iter=budget
                        )

np.random.seed(0)
clf_tune_tree.fit(X=x_train, y=y_train)

# outter evaluation by means of the test partition
y_test_pred = clf_tune_tree.predict(x_test)
print('\t outer RMSE with random search on decision tree:',metrics.mean_squared_error(y_test, y_test_pred, squared=False))
print()
# train on whole data comp_tree = clf_tune_tree.fit(X=x, y=y)


# KNN
np.random.seed(0)
param_grid = {'n_neighbors': sp_randint(2,16),
			  'p': [1, 2] }
budget = 20

np.random.seed(0)
clf_tune_knn = model_selection.RandomizedSearchCV(neighbors.KNeighborsRegressor(),
                         param_grid,
                         cv=cv_grid, 
                         n_jobs=1, verbose=1,
                         n_iter=budget
                        )

np.random.seed(0)
clf_tune_knn.fit(X=x_train, y=y_train)

y_test_pred = clf_tune_knn.predict(x_test)
print('\t outer RMSE with random search on KNN:',metrics.mean_squared_error(y_test, y_test_pred, squared=False))
print()
# train on whole data comp_knn = clf_tune_knn.fit(X=x, y=y)