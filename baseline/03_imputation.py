#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os

import datetime
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import pyampute
import pickle 
import time

from scipy.stats import mstats
from scipy.stats.mstats import winsorize
from scipy import stats
from xgboost import XGBRegressor
from sklearn import tree
from pyampute.ampute import MultivariateAmputation
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from lifelines import CoxPHFitter, WeibullFitter, WeibullAFTFitter
from datetime import datetime, date, timedelta
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from os.path import isfile, join
from sklearn.metrics import mean_absolute_error, roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from statsmodels.gam.tests.test_penalized import df_autos
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import norm
from scipy.spatial import distance
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import warnings 
warnings.filterwarnings('ignore')

from pyampute.ampute import MultivariateAmputation
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from app_hyperparameters import init_parameters_decision_tree, init_parameters_xgboost 
from app_hyperparameters import init_parameters_bayesian_ridge, init_parameters_random_forest
from app_stopping_criteria import stop_iteration
from app_uncertainty import uncertainty_sampling, multi_argmax, imputation_uncertainty
from app_uncertainty import EI
from app_init import init_truncation, init_variable_schema

import miceforest as mf 
import random

os.chdir('H:/Shared drives/CKD_Progression/')

drive = 'H'
main_path = drive + ':/Shared drives/CKD_Progression/save/qoc_cohort_ver002.csv'
data_path = drive + ':/Shared drives/CKD_Progression/data/'
docs_path = drive + ':/Shared drives/CKD_Progression/docs/'
save_path = drive + ':/Shared drives/CKD_Progression/save/'
covariates_path = docs_path + 'covariates.csv'
removecols_path = docs_path + 'remove_columns.csv'


# In[2]:


def impute(data, n_datasets, variable_schema, truncation, n_iteration = 50, 
           convergence = ['maxit', 'delta', 'early_stop'], 
           model       = ['linear', 'tree', 'forest', 'boost'], seed = 1997, verbose = True):
    
    best_parameters = None
    df_list = []
    delta_change  = []
    convergence_data = []
    hyperparameters  = {}
    for m in range(n_datasets):
        if verbose:
            print('Dataset number {:,}'.format(m + 1))
        filled_df = data.copy()
        mask = filled_df.isna()
        
        imp_order = filled_df[list(variable_schema.keys())].isna().sum().sort_values(ascending=False).index
        for to_impute in imp_order:
            np.random.seed(seed * m)
            guess = np.random.normal(loc = filled_df[to_impute].mean(), scale = filled_df[to_impute].std(), size = len(filled_df))
            filled_df[to_impute] = filled_df[to_impute].fillna(pd.Series(guess))
        it = 0
        delta = []
        post_iteration_distribution = []
        it_final, stop_condition = stop_iteration(filled_df, it, maxit = n_iteration, delta = delta, method = convergence, verbose = verbose)
        if verbose:
            print('Iteration', end = ' ')
        while stop_condition: 
            if verbose:
                print(str(it + 1), end = ', ')
            pre_it = filled_df[imp_order].copy()
            for to_impute in imp_order:
                scaler = StandardScaler()
                sampled_data = pre_it[~mask[to_impute]]
                X_train = sampled_data[variable_schema[to_impute]]
                y_train = sampled_data[to_impute]
 
                if model == 'linear':
                    estimator = BayesianRidge()   
                    if it != 0:
                        estimator.set_params(**best_parameters)

                elif model == 'tree':
                    estimator = tree.DecisionTreeRegressor()
                    if it != 0:
                        estimator.set_params(**best_parameters)
                        hyperparameters[model + f'{m}_dataset_{it}_iteration'] = best_parameters

                elif model == 'forest':
                    estimator = RandomForestRegressor()
                    if it != 0:
                        estimator.set_params(**best_parameters)
                        hyperparameters[model] = best_parameters

                elif model == 'boost':
                    estimator = XGBRegressor()
                    if it != 0:
                        estimator.set_params(**best_parameters)
                        hyperparameters[model] = best_parameters
                      
                # Expected Improvement (with bootstrapping)
                n_bootstraps = 200
                if m > 1:
                    impute_columns = variable_schema[to_impute] + [to_impute]
                    x_list = [df.loc[X_train.index, impute_columns] for df in df_list]
                    uncertainty = imputation_uncertainty(x_list)
                    mean        = np.mean(df_list[-1].loc[X_train.index, to_impute])
                    maximum     = df_list[-1].loc[X_train.index, to_impute]
                    improvement = EI(mean, uncertainty, maximum, tradeoff = 0.1)
                    query_idx, _= multi_argmax(improvement, n_instances = X_train.shape[0])
                    query_idx = [index for index in query_idx if index in X_train.index.tolist()]
                    BOOTSTRAPPED  = np.empty((n_bootstraps, filled_df.shape[0]))
                    for boot in range(n_bootstraps):
                        bootstrap_idx = np.random.choice(query_idx, size = len(query_idx), replace = True)
                        estimator.fit(X_train.loc[bootstrap_idx, :], y_train.loc[bootstrap_idx])
                        y_imputed = estimator.predict(filled_df[variable_schema[to_impute]])
                        BOOTSTRAPPED[boot, :] = y_imputed
                    y_imputed = BOOTSTRAPPED.mean(axis = 0)

                if m <= 1:
                    estimator.fit(X_train, y_train)
                    y_imputed = estimator.predict(filled_df[variable_schema[to_impute]])
                bounds = truncation[to_impute]
                y_imputed[y_imputed < bounds[0]] = bounds[0]
                y_imputed[y_imputed > bounds[1]] = bounds[1]
                y_imputed = winsorize(y_imputed, limits = (0.10, 0.10)) 
                filled_df.loc[mask[to_impute], to_impute] = y_imputed[mask[to_impute]]
                
            it = it + 1
            post_it = filled_df[imp_order].copy()
            if convergence == 'delta':
                delta_val = ((post_it-pre_it)**2).sum() / ((post_it) ** 2).sum()
            elif convergence == 'early_stop':
                delta_val = np.sqrt((((post_it - pre_it)**2)/len(post_it)).sum())
            else:
                delta_val = pd.Series(np.NaN)
            delta.append(delta_val)
            post_iteration_distribution.append(post_it)
            it_final, stop_condition = stop_iteration(post_it, it, maxit = n_iteration, delta = delta, 
                                                      method = convergence, verbose = verbose)

            penalty = int(np.array((pd.DataFrame(delta).median(axis = 0))).max())
            if penalty <= 1:
                penalty = 2
        
            if stop_condition:
                if model == 'tree':
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    decision_tree_estimator = tree.DecisionTreeRegressor()
                    randomized_search = RandomizedSearchCV(decision_tree_estimator, param_distributions = init_parameters_decision_tree(), 
                                                           cv = 10, n_iter = 1, n_jobs = 1)
                    
                    randomized_search.fit(X_train, y_train)
                    best_parameters = randomized_search.best_params_

                elif model == 'boost':
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    xgboost_estimator = XGBRegressor(random_state = 42)
                    randomized_search = RandomizedSearchCV(xgboost_estimator, param_distributions = init_parameters_xgboost(), 
                                                           cv = 10, n_iter = 2, n_jobs = -1)
                    randomized_search.fit(X_train, y_train)
                    best_parameters = randomized_search.best_params_

                elif model == 'linear':
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    linear_estimator  = BayesianRidge()
                    randomized_search = RandomizedSearchCV(linear_estimator, param_distributions = init_parameters_bayesian_ridge(), 
                                                           cv = 10, n_iter = 2, n_jobs = -1)
                    randomized_search.fit(X_train, y_train)
                    best_parameters = randomized_search.best_params_      

                elif model == 'forest':
                    scaler = StandardScaler()
                    X_train = scaler.fit_transform(X_train)
                    random_forest_estimator  = RandomForestRegressor()
                    randomized_search = RandomizedSearchCV(random_forest_estimator, 
                                                           param_distributions = init_parameters_random_forest(), 
                                                           cv = 10, n_iter = 2, n_jobs = 10)
                    randomized_search.fit(X_train, y_train)
                    best_parameters = randomized_search.best_params_                 
        df_list.append(filled_df)
        delta_change.append(pd.concat(delta, axis = 'columns'))
        convergence_data.append(post_iteration_distribution)
    result = {'imputed_data': df_list, 'convergence_data': convergence_data, 'iteration_delta': delta_change}
    return result, hyperparameters


# In[3]:


columns_impute = ['BMI', 'BW', 'HIGH', 'Lipid_HDL', 'Lipid_LDL', 'Chem_glucose', 'Chem_HbA1C', 'Renal_Uric_acid', 
                  'Lipid_Cholesterol', 'Renal_Serum_creatinine', 'Lipid_Triglyceride', 'age', 'Renal_eGFR', 'CVD']

main_data = pd.read_csv(save_path + 'qoc_cohort_ver002.csv')
main_data = main_data.rename(columns = {'height': 'HIGH'})
main_data = main_data[columns_impute]


# In[ ]:


def impute_loop(data, model, convergence, path):
    name = model + "_" + convergence
    path = path+"/"+name+"_40D_10I_ver2.pickle"
    imputed_data, hyperparameters = impute(data = data, n_datasets = 40, variable_schema = init_variable_schema(), 
                          truncation = init_truncation(), n_iteration = 10, convergence = convergence, 
                          seed = 1996, model = model, verbose = True)
    with open(path, 'wb') as handle:
        pickle.dump(imputed_data, handle, protocol = pickle.HIGHEST_PROTOCOL)
    return hyperparameters

models = ['tree']
records = []
for model in models:
    for convergence in ['early_stop']:
        print(model, convergence)
        print('{} model with {} convergence'.format(model, convergence), end = ', ')
        start = time.time()
        hyperparameters = impute_loop(data = main_data, model = model, convergence = convergence, 
                                      path = 'H:/Shared drives/CKD_Progression/result/imputation/')
        stop = time.time()
        print('time taken to impute {:.4f} seconds'.format(stop - start))
        records.append([model, convergence, 'sequential',  stop - start])

temp = pd.Series(records)
temp.columns = ['model', 'convergence', 'datasets', 'time_taken']


# In[4]:


def impute_loop(data, model, convergence, path):
    name = model + "_" + convergence
    path = path+"/"+name+"_40D_10I.pickle"
    imputed_data, hyperparameters = impute(data = data, n_datasets = 40, variable_schema = init_variable_schema(), 
                          truncation = init_truncation(), n_iteration = 10, convergence = convergence, 
                          seed = 1996, model = model, verbose = True)
    with open(path, 'wb') as handle:
        pickle.dump(imputed_data, handle, protocol = pickle.HIGHEST_PROTOCOL)
    return hyperparameters

models = ['tree']
records = []
for model in models:
    for convergence in ['early_stop']:
        print(model, convergence)
        print('{} model with {} convergence'.format(model, convergence), end = ', ')
        start = time.time()
        hyperparameters = impute_loop(data = main_data, model = model, convergence = convergence, 
                                      path = 'H:/Shared drives/CKD_Progression/result/imputation/')
        stop = time.time()
        print('time taken to impute {:.4f} seconds'.format(stop - start))
        records.append([model, convergence, 'sequential',  stop - start])

temp = pd.Series(records)
temp.columns = ['model', 'convergence', 'datasets', 'time_taken']

