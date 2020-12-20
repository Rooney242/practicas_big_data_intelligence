import pandas as pd
import numpy as np
import sys
import statistics as st
from sklearn import preprocessing, model_selection, tree, neighbors, metrics
from scipy.stats import uniform, randint as sp_randint
from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical
import optuna


#MAIN PARAMETERS FOR THE ASSIGNMENT
budget = 30
random_state = 0
verbose = 0

#PARAMETERS FOR THE HYPER-PARAMETER TUNNING
min_max_depth = 2
max_max_depth = 16
min_n_neigbors = 1
max_n_neigbors = 16

#Dataframes with all the information of each model
summary = {
    'tree': pd.DataFrame(columns=['Time', 'Score (RMSE)', 'Min. samples split', 'Criterion', 'Max. depth']),
    'knn': pd.DataFrame(columns=['Time', 'Score (RMSE)', 'N. neighbors', 'Weights', 'P'])
}

#Loading data
data = pd.read_csv("kaggleCompetition.csv")
data = data.values

#Splitting data in the one used for training and the one used for the competition
x = data[0:1460, :-1]
y = data[0:1460, -1] 
x_comp = data[1460:,:-1] 
y_comp = data[1460:,-1]

#standardize input attributes.
scaler = preprocessing.StandardScaler().fit(x) 
x = scaler.transform(x)
x_comp = scaler.transform(x_comp)

#split in train/test sets using holdout 3/4 for training, 1/4 for testing
x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, train_size=0.75, random_state=0)

#hyperparams evaluated by 2-fold CV (inner evaluation)
cv_grid = model_selection.KFold(n_splits=2, shuffle=True, random_state=random_state)


###3.1 Evaluate a DecisionTreeRegressor and a KNeighborsRegressor with default hyper-parameters.
print('##### DEFAULT PARAMETERS #####')
#3.1.1 Decision Tree
np.random.seed(random_state)
tree_default = tree.DecisionTreeRegressor(random_state=random_state)
scores = -model_selection.cross_val_score(tree_default, x_train, y_train, scoring='neg_root_mean_squared_error', cv=cv_grid)

summary['tree'] = summary['tree'].append(pd.Series({
    'Time': 0, 
    'Score (RMSE)': scores.mean(),
    'Min. samples split': 2, 
    'Criterion': 'mse', 
    'Max. depth': 'None'
    },
    name='default'))

#3.1.2 K Nearest neighbours
np.random.seed(random_state)
knn_default = neighbors.KNeighborsRegressor()
scores = -model_selection.cross_val_score(knn_default, x_train, y_train, scoring='neg_root_mean_squared_error', cv=cv_grid) 

summary['knn'] = summary['knn'].append(pd.Series({
    'Time': 0, 
    'Score (RMSE)': scores.mean(), 
    'N. neighbors': 5, 
    'Weights': 'uniform', 
    'P': 2
    }, 
    name='default'))

###3.2 Random search for Decission Tree hyper-parameter tunning
print('##### RANDOM SEARCH #####')
np.random.seed(random_state)
param_grid = {
    'min_samples_split': uniform(0, 1),
    'criterion': ['mse','friedman_mse'], 
    'max_depth': sp_randint(min_max_depth, max_max_depth)
}
tree_random_search = model_selection.RandomizedSearchCV(
    tree.DecisionTreeRegressor(random_state=random_state), 
    param_grid,
    scoring='neg_root_mean_squared_error',
    cv=cv_grid, 
    verbose=verbose,
    n_iter=budget
    )
tree_random_search.fit(X=x_train, y=y_train)

summary['tree'] = summary['tree'].append(pd.Series({
    'Time': 0, 
    'Score (RMSE)': -tree_random_search.best_score_,
    'Min. samples split': tree_random_search.best_params_['min_samples_split'], 
    'Criterion': tree_random_search.best_params_['criterion'], 
    'Max. depth': tree_random_search.best_params_['max_depth']
    },
    name='random_search'))

###3.3 Random search for K Nearest Neighbours hyper-parameter tunning
np.random.seed(random_state)
param_grid = {
    'n_neighbors': sp_randint(min_n_neigbors, max_n_neigbors),
    'weights': ['uniform', 'distance'],
    'p': [1, 2]
}

knn_random_search = model_selection.RandomizedSearchCV(
    neighbors.KNeighborsRegressor(), 
    param_grid,
    scoring='neg_root_mean_squared_error',
    cv=cv_grid, 
    verbose=verbose,
    n_iter=budget
    )
knn_random_search.fit(X=x_train, y=y_train)

summary['knn'] = summary['knn'].append(pd.Series({
    'Time': 0, 
    'Score (RMSE)': -knn_random_search.best_score_, 
    'N. neighbors': knn_random_search.best_params_['n_neighbors'], 
    'Weights': knn_random_search.best_params_['weights'], 
    'P': knn_random_search.best_params_['p']
    }, 
    name='random_search'))


###3.4 Skopt (bayesian) hyper-parameter tunning
print('##### SKOPT #####')
#3.4.1 Decission trees
np.random.seed(random_state)
param_grid = {
    'min_samples_split': Real(0+sys.float_info.min, 1),
    'criterion': Categorical(['mse','friedman_mse']), 
    'max_depth': Integer(min_max_depth, max_max_depth)
}
tree_skopt = BayesSearchCV(tree.DecisionTreeRegressor(random_state=random_state), 
    param_grid,
    cv=cv_grid,    
    verbose=verbose,
    scoring='neg_root_mean_squared_error',
    n_iter=budget
    )
tree_skopt.fit(X=x_train, y=y_train)

summary['tree'] = summary['tree'].append(pd.Series({
    'Time': 0, 
    'Score (RMSE)': -tree_skopt.best_score_,
    'Min. samples split': tree_skopt.best_params_['min_samples_split'], 
    'Criterion': tree_skopt.best_params_['criterion'], 
    'Max. depth': tree_skopt.best_params_['max_depth']
    },
    name='skopt'))

#3.4.1 K Nearest neighbours
np.random.seed(random_state)
param_grid = {
    'n_neighbors': Integer(min_n_neigbors, max_n_neigbors),
    'weights': Categorical(['uniform', 'distance']),
    'p': Categorical([1, 2])
}
knn_skopt = BayesSearchCV(neighbors.KNeighborsRegressor(), 
    param_grid,
    cv=cv_grid,    
    verbose=verbose,
    scoring='neg_root_mean_squared_error',
    n_iter=budget
    )
knn_skopt.fit(X=x_train, y=y_train)

summary['knn'] = summary['knn'].append(pd.Series({
    'Time': 0, 
    'Score (RMSE)': -knn_skopt.best_score_, 
    'N. neighbors': knn_skopt.best_params_['n_neighbors'], 
    'Weights': knn_skopt.best_params_['weights'], 
    'P': knn_skopt.best_params_['p']
    }, 
    name='skopt'))

###3.5 Optuna (bayesian) hyper-parameter tunning
print('##### OPTUNA #####')
optuna.logging.set_verbosity(verbose)
#3.5.1 Decission trees
np.random.seed(random_state)
def tree_objective(trial):
    min_samples_split = trial.suggest_uniform('min_samples_split', 0+sys.float_info.min, 1)
    criterion = trial.suggest_categorical('criterion', ['mse','friedman_mse'])
    max_depth = trial.suggest_int('max_depth', min_max_depth, max_max_depth)

    clf = tree.DecisionTreeRegressor(
        random_state=random_state,
        min_samples_split=min_samples_split,
        criterion=criterion,
        max_depth=max_depth)

    scores = -model_selection.cross_val_score(clf, x_train, y_train,
        cv=cv_grid,
        verbose=verbose,
        scoring='neg_root_mean_squared_error')

    return scores.mean()

tree_optuna = optuna.create_study(direction='minimize')
tree_optuna.optimize(tree_objective, n_trials=budget)

summary['tree'] = summary['tree'].append(pd.Series({
    'Time': 0, 
    'Score (RMSE)': tree_optuna.best_value,
    'Min. samples split': tree_optuna.best_params['min_samples_split'], 
    'Criterion': tree_optuna.best_params['criterion'], 
    'Max. depth': tree_optuna.best_params['max_depth']
    },
    name='optuna'))
#3.5.2 K Nearest Neighbours
np.random.seed(random_state)
def knn_objective(trial):
    n_neighbors = trial.suggest_int('n_neighbors', min_n_neigbors, max_n_neigbors)
    weights = trial.suggest_categorical('weights', ['uniform','distance'])
    p = trial.suggest_categorical('p', [1, 2])

    clf = neighbors.KNeighborsRegressor(
        n_neighbors=n_neighbors,
        weights=weights,
        p=p)

    scores = -model_selection.cross_val_score(clf, x_train, y_train,
        cv=cv_grid,
        verbose=verbose,
        scoring='neg_root_mean_squared_error')

    return scores.mean()

knn_optuna = optuna.create_study(direction='minimize')
knn_optuna.optimize(knn_objective, n_trials=budget)

summary['knn'] = summary['knn'].append(pd.Series({
    'Time': 0, 
    'Score (RMSE)': knn_optuna.best_value, 
    'N. neighbors': knn_optuna.best_params['n_neighbors'], 
    'Weights': knn_optuna.best_params['weights'], 
    'P': knn_optuna.best_params['p']
    }, 
    name='optuna'))

print(summary['tree'])
print(summary['knn'])

###3.6 Determine the best model from its inner evaluation
#Decission Tree
best_tree_model = summary['tree']['Score (RMSE)'].idxmin()
print('The Decision Tree Regressor with best model is:', best_tree_model)

#K Nearest Neighbors
best_knn_model = summary['knn']['Score (RMSE)'].idxmin()
print('The K Nearest Neighbors Regressor with best model is:', best_knn_model)

if summary['tree'].loc[best_tree_model]['Score (RMSE)'] < summary['tree'].loc[best_knn_model]['Score (RMSE)']:
    print('The best model is Decision Tree Regressor with {}'.format(best_tree_model))
    best_model = tree.DecisionTreeRegressor(
        random_state=random_state,
        min_samples_split=summary['tree'].loc[best_tree_model]['Min. samples split'] ,
        criterion=summary['tree'].loc[best_tree_model]['Criterion'],
        max_depth=summary['tree'].loc[best_tree_model]['Max. depth'])
else:
    print('The best model is K Nearest Neighbors Regressor with {}'.format(best_knn_model))
    best_model = neighbors.KNeighborsRegressor(
        n_neighbors=summary['knn'].loc[best_knn_model]['N. neighbors'] ,
        weights=summary['knn'].loc[best_knn_model]['Weights'],
        p=summary['knn'].loc[best_knn_model]['P'])


###3.7 Performance estimation
best_model.fit(x_train, y_train)
best_model_predict = best_model.predict(x_test)
print('Best Model performance at competition:')
print('\t MSE: {:.4f} (should be better than the dummy predictor MSE {:.4f})'.format(
    metrics.mean_squared_error(y_test, best_model_predict),
    metrics.mean_squared_error(y_test, [y_test.mean() for i in range(len(y_test))])))
print('\t R square: {:.4f} (should be better than the dummy predictor R square {:.4f})'.format(
    metrics.r2_score(y_test, best_model_predict),
    metrics.r2_score(y_test, [y_test.mean() for i in range(len(y_test))])))

#3.8 Final model train
best_model.fit(x, y)