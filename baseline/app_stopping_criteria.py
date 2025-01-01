import numpy as np 

def stop_iteration(it_final, it, maxit, delta, 
                   method = ['maxit', 'delta', 'early_stop'], verbose = False):
    more_it = False
    if method == 'maxit':
        more_it = it < maxit
    elif method == 'delta':
        if it<maxit:
            if len(delta)<2:
                more_it = True
            else:
                more_it = all(delta[-1] < delta[-2])
    elif method == 'early_stop':
        if it< maxit:
            if len(delta) < 5:
                more_it = True
            else:
                more_it = all([all([i<1e-2 for i in j]) for j in delta[-5:]]) == False
    if more_it is False:
        if verbose:
            print('stopping at {:,} iteration(s)'.format(it))
    return it_final, more_it

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return filtered_df