import numpy as np 

def init_parameters_decision_tree():
    parameters = {
        'criterion': ['absolute_error', 'friedman_mse', 'squared_error'],
        'splitter': ['best', 'random'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }
    return parameters

def init_parameters_xgboost():
    parameters = {
        'n_estimators': [5, 10, 50],
        'learning_rate': [0.01, 0.1, 0.3], 
        'max_depth': [3, 6, 9],
        'min_child_weight': [1, 3, 5],  
        'subsample': [0.7, 0.8, 0.9], 
        'colsample_bytree': [0.7, 0.8, 0.9],  
        'reg_lambda': [0, 1, 10],  
        'reg_alpha': [0, 1, 10],  
        'gamma': [0, 0.1, 0.3],  
        'objective': ['reg:squarederror'], 
        'eval_metric': ['rmse', 'mae'], 
    }
    return parameters

def init_parameters_bayesian_ridge():
    parameters = { 
        'alpha_1':  [1e-6, 1e-7, 1e-8], 
        'alpha_2':  [1e-6, 1e-7, 1e-8],  
        'lambda_1': [1e-6, 1e-7, 1e-8], 
        'lambda_2': [1e-6, 1e-7, 1e-8]}
    return parameters

def init_parameters_random_forest():
    param_grid = { 
                'n_estimators': [10, 20], 
                'criterion': ['squared_error', 'friedman_mse'],
                'max_depth': np.arange(1, 12),
                'min_samples_split': np.arange(2, 12)}
    return param_grid