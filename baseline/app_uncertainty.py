import numpy as np 
from sklearn.exceptions import NotFittedError
from scipy.special import ndtr
from sklearn.metrics import mean_absolute_error, roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from statsmodels.gam.tests.test_penalized import df_autos
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import norm
from scipy.spatial import distance
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df

def uncertainty_sampling(X_pool, model, n_instances = 10):
    probas = model.predict_proba(X_pool)
    uncertainty = 1 - np.abs(probas[:, 0] - probas[:, 1])
    return np.argsort(uncertainty)[-n_instances:]

def UCB(mean, std, beta):
    return mean + beta*std

def PI(mean, std, max_val, tradeoff = 0.00001):
    return ndtr((mean - max_val)/std)

def EI(mean, std, max_val, tradeoff = 0.00001):
    z = (mean - max_val - tradeoff) / std
    return (mean - max_val - tradeoff)*ndtr(z) + std*norm.pdf(z)

def optimizer_UCB(optimizer, X, beta: float = 1) -> np.ndarray:
    try:
        mean, std = optimizer.predict(X, return_std=True)
        mean, std = mean.reshape(-1, ), std.reshape(-1, )
    except NotFittedError:
        mean, std = np.zeros(shape = (X.shape[0], 1)), np.ones(shape = (X.shape[0], 1))
    return UCB(mean, std, beta)

def optimizer_PI(optimizer, X, tradeoff):
    try:
        mean, std = optimizer.predict(X, return_std=True)
        mean, std = mean.reshape(-1, ), std.reshape(-1, )
    except NotFittedError:
        mean, std = np.zeros(shape = (X.shape[0], 1)), np.ones(shape=(X.shape[0], 1))
    return PI(mean, std, optimizer.y_max, tradeoff)

def optimizer_EI(optimizer, X, tradeoff):
    try:
        mean, std = optimizer.predict(X, return_std=True)
        mean, std = mean.reshape(-1, ), std.reshape(-1, )
    except NotFittedError:
        mean, std = np.zeros(shape=(X.shape[0], 1)), np.ones(shape=(X.shape[0], 1))

    return EI(mean, std, optimizer.y_max, tradeoff)

def max_UCB(optimizer, X, beta: float = 1, n_instances: int = 1) -> np.ndarray:
    ucb = optimizer_UCB(optimizer, X, beta = beta)
    return multi_argmax(ucb, n_instances = n_instances)

def max_EI(optimizer, X, tradeoff, n_instances):
    ei = optimizer_EI(optimizer, X, tradeoff = tradeoff)
    return multi_argmax(ei, n_instances = n_instances)

def multi_argmax(values: np.ndarray, n_instances: int = 1) -> np.ndarray:
    assert n_instances <= values.shape[0]
    max_idx = np.argpartition(-values, n_instances-1, axis = 0)[:n_instances]
    max_idx = [index for index in max_idx if index in values.index.tolist()]
    return max_idx, values[max_idx]

def imputation_uncertainty(imputed_list):
    delta = np.var(imputed_list, axis = 0, ddof = 1)
    delta = np.sum(delta, axis = 1)
    return np.sqrt(delta)

def calculate_mean(imputed_list):
    delta = np.mean(imputed_list, axis = 0)
    return delta

def shuffled_argmax(values: np.ndarray, n_instances: int = 1) -> np.ndarray:
    assert n_instances <= values.shape[0]
    shuffled_idx = np.random.permutation(len(values))
    shuffled_values = values[shuffled_idx]
    sorted_query_idx = np.argsort(shuffled_values, kind = 'mergesort')[
        len(shuffled_values)-n_instances:]
    query_idx = shuffled_idx[sorted_query_idx]
    return query_idx, values[query_idx]

def shuffled_argmin(values: np.ndarray, n_instances: int = 1) -> np.ndarray:
    assert n_instances <= values.shape[0]
    shuffled_idx = np.random.permutation(len(values))
    shuffled_values = values[shuffled_idx]
    sorted_query_idx = np.argsort(shuffled_values, kind = 'mergesort')[:n_instances]
    query_idx = shuffled_idx[sorted_query_idx]
    return query_idx, values[query_idx]

def shuffled_argmax_threshold(values: np.ndarray, threshold: float = 0.5, n_instances: int = 1) -> np.ndarray:
    assert n_instances <= values.shape[0]
    shuffled_idx = np.random.permutation(len(values))
    shuffled_values = values[shuffled_idx]
    filtered_idx = np.where(shuffled_values > threshold)[0][n_instances:]
    query_idx = shuffled_idx[filtered_idx]
    return query_idx, values[query_idx]

def shuffled_argmin_threshold(values: np.ndarray, threshold: float = 0.00001, n_instances: int = 1) -> np.ndarray:
    assert n_instances <= values.shape[0]
    shuffled_idx = np.random.permutation(len(values))
    shuffled_values = values[shuffled_idx]
    filtered_idx = np.where(shuffled_values < threshold)[0][:n_instances]
    query_idx = shuffled_idx[filtered_idx]
    return query_idx, values[query_idx]