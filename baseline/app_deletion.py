def univariate_coxph(df, pathway, save = False):
    HAZARD, CONF_INT, PVAL, SERROR, EVENTS_OBS, EVENTS_TOT = [], [], [], [], [], []
    for covariate in tqdm(covariates):
        try:
            cph_model = generate_coxph(df, covariate, pathway)
            hazd, confidence_interval, pvalue, ster, event_obs, event_tot = coxph_statistics(cph_model)
            HAZARD.append(hazd)
            CONF_INT.append(confidence_interval)
            PVAL.append(pvalue)
            SERROR.append(ster)
            EVENTS_OBS.append(event_obs)
            EVENTS_TOT.append(event_tot)
        except Exception as e:
            HAZARD.append(np.nan)
            CONF_INT.append(np.nan)
            PVAL.append(np.nan)
            SERROR.append(np.nan)
            EVENTS_OBS.append(np.nan)
            EVENTS_TOT.append(np.nan)
    data = (covariates, HAZARD, CONF_INT, PVAL, SERROR, EVENTS_OBS, EVENTS_OBS)
    univariate_covariate = pd.DataFrame(data).T
    univariate_covariate = univariate_covariate.rename(columns = {0: 'covariate', 1: 'hazard', 2:'95CI', 3: 'pvalue', 4:'SE', 5: 'observe', 6: 'total_patients'})
    # univariate_covariate = pd.merge(univariate_covariate, order_covariates, on = 'covariate', how = 'inner')
    # univariate_covariate = univariate_covariate.sort_values(['order'])
    # univariate_covariate = replace_covariate_labels(univariate_covariate)
    # univariate_covariate = replace_pvalue(univariate_covariate)
    return univariate_covariate