import numpy as np
import pandas as pd
import os

import datetime
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
import pickle 
import time
import ast 

from scipy.stats import mstats
from scipy.stats.mstats import winsorize
from scipy import stats
from sklearn import tree
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
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
from app_transition_dict import get_transition_dict, get_transition_code
from app_init import get_multi_state_covariates, get_multi_state_cov_quartiles
from app_init import replace_covariate_labels, replace_pvalue, get_variables_cox
import warnings 
warnings.filterwarnings('ignore')

drive = 'G'
main_path = drive + ':/Shared drives/CKD_Progression/data/CKD_COHORT_Jan2010_Mar2024_v3.csv'
data_path = drive + ':/Shared drives/CKD_Progression/data/'
docs_path = drive + ':/Shared drives/CKD_Progression/docs/'
save_path = drive + ':/Shared drives/CKD_Progression/save/'
resu_path = drive + ':/Shared drives/CKD_Progression/result/'
covariates_path = docs_path + 'covariates.csv'
removecols_path = docs_path + 'remove_columns.csv'

def generate_df_continuous(df, variables, q = 3):
    bins_dict = {} 
    for variable, prefix, column_name in variables:
        df[column_name], bins = pd.qcut(
            df[variable],
            q = q,
            labels = False,
            duplicates = 'drop',
            retbins = True)
        bins_dict[variable] = bins
        dummies = pd.get_dummies(df[column_name], prefix = prefix)
        df = pd.concat([df, dummies], axis = 1)
    return df, bins_dict

def generate_df_continuous_predefined(df, variables, get_columns = False):
    for variable, prefix, column_name, bins, labels in variables:
        df[column_name] = pd.cut(df[variable], bins = bins, labels = labels, right = False)

        if get_columns:
            dummies = pd.get_dummies(df[column_name], prefix = prefix)
            df = pd.concat([df, dummies], axis = 1)
    return df

def get_first_dates():
    heart_failure = pd.read_excel(docs_path + 'HF_FIRSTDATE_2010_2023.xlsx') ['ENC_HN'].unique().tolist()
    hypertension  = pd.read_excel(docs_path + 'HTN_FIRSTDATE_2010_2023.xlsx')['ENC_HN'].unique().tolist()
    diabetes = pd.read_csv(docs_path + 'DM_FIRSTDATE_2010_2023.csv')['ENC_HN'].unique().tolist()
    atrialfb = pd.read_csv(docs_path + 'AF_FIRSTDATE_2010_2023.csv')['ENC_HN'].unique().tolist()
    return heart_failure, hypertension, diabetes, atrialfb

def merge_comorbidity(df, comorbidity, disease_code):
    disease_column = disease_code.upper()
    df[disease_column] = df['ENC_HN'].isin(comorbidity).astype(int)
    return df

def load_dataset(get_columns = False):
    covariates, variables = get_multi_state_cov_quartiles(), get_variables_cox()
    order_covariates = pd.read_csv(docs_path + 'cox_covariates.csv')
    model_vars = ['ENC_HN', 'transition', 'fr', 'to', 'status', 'tstart', 'tstops', 'time']
    heart_failure, hypertension, diabetes, atrialfb = get_first_dates()
    long_df = pd.read_csv(save_path + 'multi_state_long_ver013.csv')
    long_df['gender']  = long_df['gender'].replace('M', 1).replace('F', 0)
    long_df['pathway'] = long_df['fr'] + '_to_' + long_df['to']
    long_df = generate_df_continuous_predefined(long_df, variables, get_columns = get_columns)
    long_df['statin']  = long_df[['statinhydro', 'statinlipo']].max(axis = 1)
    long_df['raas']    = long_df[['arb', 'acei']].max(axis = 1)
    long_df = long_df.drop(columns = ['statinhydro', 'statinlipo'])
    long_df = merge_comorbidity(long_df, heart_failure, 'hf')
    long_df = merge_comorbidity(long_df, diabetes,      'dm')
    long_df = merge_comorbidity(long_df, atrialfb,      'af')
    return covariates, order_covariates, long_df 