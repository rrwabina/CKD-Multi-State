def init_imputation_columns():
    columns_impute = ['BMI', 
                      'BW', 
                      'HIGH', 
                      'Lipid_HDL', 
                      'Lipid_LDL', 
                      'Chem_glucose', 
                      'Chem_HbA1C', 
                      'Renal_Uric_acid', 
                      'Lipid_Cholesterol', 
                      'Renal_Serum_creatinine', 
                      'Lipid_Triglyceride', 
                      'age', 
                      'Renal_eGFR', 
                      'CVD']
    return columns_impute

def init_truncation():
    truncation = {
        'age':              [ 18, 130],
        'Renal_eGFR':       [  1, 150],
        'HIGH':             [120, 200],
        'BW':               [ 20, 200],
        'Lipid_HDL':        [20, 800],
        'Lipid_LDL':        [10, 160],
        'Chem_glucose':     [20, 2000],
        'Chem_HbA1C':       [4, 20],
        'Renal_Uric_acid':  [2, 40],
        'Lipid_Cholesterol':[30, 1500],
        'Renal_Serum_creatinine':   [0.1, 40],
        'Lipid_Triglyceride':       [30, 10000]}
    return truncation


def init_variable_schema():
    variable_schema = {
        'BW': ['Lipid_Cholesterol', 'Renal_Serum_creatinine', 'Chem_glucose', 'Lipid_HDL', 'Lipid_LDL', 'Lipid_Triglyceride', 'HIGH'],
        'HIGH': ['Lipid_Cholesterol', 'Renal_Serum_creatinine', 'Chem_glucose', 'Lipid_HDL', 'Lipid_LDL', 'Lipid_Triglyceride', 'BW'],
        'Lipid_HDL': ['Lipid_Cholesterol', 'Chem_glucose', 'Lipid_LDL', 'Lipid_Triglyceride', 'BW'],
        'Lipid_LDL': ['Lipid_Cholesterol', 'Renal_Serum_creatinine', 'Lipid_HDL', 'Lipid_Triglyceride', 'BW'],
        'Chem_glucose': ['Lipid_Cholesterol', 'Lipid_Triglyceride', 'Renal_Serum_creatinine', 'BW', 'Lipid_HDL'],
        'Chem_HbA1C': ['Lipid_Cholesterol', 'Renal_Serum_creatinine', 'Chem_glucose', 'Lipid_LDL', 'Lipid_HDL', 'Lipid_Triglyceride'],
        'Renal_Uric_acid': ['Lipid_Cholesterol', 'Renal_Serum_creatinine', 'Chem_glucose', 'Lipid_HDL', 'Lipid_LDL', 'Lipid_Triglyceride', 'BW'],
        'Lipid_Cholesterol': ['Lipid_HDL', 'Lipid_LDL', 'Lipid_Triglyceride', 'HIGH', 'BW'],
        'Renal_Serum_creatinine': ['Lipid_Cholesterol', 'Chem_glucose', 'Lipid_HDL', 'Lipid_LDL', 'Lipid_Triglyceride', 'BW'],
        'Lipid_Triglyceride': ['Lipid_Cholesterol', 'Chem_glucose', 'Lipid_HDL', 'Lipid_LDL', 'BW']}
    return variable_schema


def get_multi_state_covariates():
    covariates = [
        'BMI','BW', 'HIGH', 'Lipid_HDL', 'Lipid_LDL', 'Chem_glucose', 'Chem_HbA1C',
       'Renal_Uric_acid', 'Lipid_Cholesterol', 'Renal_Serum_creatinine',
       'Lipid_Triglyceride', 'age', 'Renal_eGFR', 'gender', 'ANTI_PL',
       'PHOS_BINDER', 'dpp4', 'glp1', 'sglt2', 'acei', 'arb', 'bb',
       'statinhydro', 'statinlipo', 'HT', 'PVD', 'stroke', 'DLP', 'Gout']
    return covariates

def get_multi_state_cov_quartiles():
    covariates = ['gender', 'ANTI_PL', 'PHOS_BINDER', 'dpp4', 'glp1', 'sglt2', 'raas', 'bb', 'HT', 'PVD', 'HF', 'DM', 'AF', 'DLP', 'Gout', 
                  'bin_age', 'bin_bmi', 'bin_hdl', 'bin_ldl', 'bin_cho', 'bin_tri', 'bin_glu', 'bin_hba', 'bin_rua', 'statin']
    return covariates

def get_modeling_vars():
    variables = ['ENC_HN', 'transition', 'fr', 'to', 'status', 'tstart', 'tstops', 'time']
    return variables

def replace_covariate_labels(data):
    data['covariate'] = data['covariate'].replace('gender', 'Gender')
    data['covariate'] = data['covariate'].replace('ANTI_PL', 'Anti-platelet drugs')
    data['covariate'] = data['covariate'].replace('PHOS_BINDER', 'Phosphate binder')
    data['covariate'] = data['covariate'].replace('dpp4',  'DPP-4 inhibitor')
    data['covariate'] = data['covariate'].replace('glp1',  'GLP-1 inhibitor')
    data['covariate'] = data['covariate'].replace('sglt2', 'SGLT-2 inhibitor')
    data['covariate'] = data['covariate'].replace('acei', 'ACEI')
    data['covariate'] = data['covariate'].replace('arb', 'ARB')
    data['covariate'] = data['covariate'].replace('bb', 'Beta Blockers')
    data['covariate'] = data['covariate'].replace('statinhydro', 'Hydrophilic statin')
    data['covariate'] = data['covariate'].replace('statinlipo', 'Lipophilic statin')
    data['covariate'] = data['covariate'].replace('HT',  'Hypertension')
    data['covariate'] = data['covariate'].replace('PVD', 'Peripheral Vascular Disease')
    data['covariate'] = data['covariate'].replace('stroke', 'Stroke')
    data['covariate'] = data['covariate'].replace('DLP', 'Dyslipidemia')
    data['covariate'] = data['covariate'].replace('Gout', 'Gout')
    data['covariate'] = data['covariate'].replace('T2DM', 'Type 2 DM')
    data['covariate'] = data['covariate'].replace('Lipid_HDL', 'HDL')
    data['covariate'] = data['covariate'].replace('Lipid_LDL', 'LDL')
    data['covariate'] = data['covariate'].replace('Chem_glucose', 'Glucose')
    data['covariate'] = data['covariate'].replace('Renal_Uric_acid', 'Uric acid')
    data['covariate'] = data['covariate'].replace('Chem_HbA1C', 'HbA1C')
    data['covariate'] = data['covariate'].replace('Lipid_Cholesterol', 'Cholesterol')
    data['covariate'] = data['covariate'].replace('Renal_Serum_creatinine', 'Serum creatinine')
    data['covariate'] = data['covariate'].replace('Lipid_Triglyceride', 'Triglyceride')
    data['covariate'] = data['covariate'].replace('age', 'Age')
    data = data.sort_values(['covariate'])
    return data

def replace_pvalue(data):
    data['pvalue'] = [0.0 if (value <= 0.001) else value for value in data['pvalue']]
    data['pvalue'] = data['pvalue'].replace(0.0, '<0.001')
    return data

def get_variables_cox():
    variables = [
        ('age', 'age', 'bin_age', [0, 60, float('inf')], ['less60', 'geq60']),
        ('BMI', 'bmi', 'bin_bmi', [0, 18.5, 23, float('inf')], ['under', 'normal', 'over']),
        ('Lipid_HDL', 'hdl', 'bin_hdl', [0, 40, float('inf')], ['low', 'normal']),
        ('Lipid_LDL', 'ldl', 'bin_ldl', [0, 100, float('inf')], ['normal', 'high']),
        ('Lipid_Cholesterol', 'cho', 'bin_cho', [0, 200, float('inf')], ['normal', 'high']),
        ('Lipid_Triglyceride', 'tri', 'bin_tri', [0, 150, float('inf')], ['normal', 'high']),
        ('Chem_glucose', 'glu', 'bin_glu', [0, 100, 126, float('inf')], ['normal', 'impaired', 'high']),
        ('Chem_HbA1C', 'hba', 'bin_hba', [0, 5.7, 6.5, float('inf')], ['normal', 'prediabetes', 'high']),
        ('Renal_Uric_acid', 'rua', 'bin_rua', [0, 7.0, float('inf')], ['normal', 'hyper'])]
    return variables
