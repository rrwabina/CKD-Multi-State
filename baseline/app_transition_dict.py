
def get_transition_dict():
    transition_dict = {
        'ckd3a_to_ckd3b_months': ('CKD3A_date', 'CKD3B_date'),
        'ckd3a_to_ckd04_months': ('CKD3A_date', 'CKD4_date'),
        'ckd3a_to_ckd5a_months': ('CKD3A_date', 'CKD5A_date'),
        'ckd3a_to_ckd5b_months': ('CKD3A_date', 'CKD5B_date'),
        'ckd3a_to_cvd00_months': ('CKD3A_date', 'CVD_date'),
        'ckd3a_to_death_months': ('CKD3A_date', 'DEAD_date'),
        
        'ckd3b_to_ckd04_months': ('CKD3B_date', 'CKD4_date'),
        'ckd3b_to_ckd5a_months': ('CKD3B_date', 'CKD5A_date'),
        'ckd3b_to_ckd5b_months': ('CKD3B_date', 'CKD5B_date'),
        'ckd3b_to_cvd00_months': ('CKD3B_date', 'CVD_date'),
        'ckd3b_to_death_months': ('CKD3B_date', 'DEAD_date'),
        
        'ckd04_to_ckd5a_months': ('CKD4_date', 'CKD5A_date'),
        'ckd04_to_ckd5b_months': ('CKD4_date', 'CKD5B_date'),
        'ckd04_to_cvd00_months': ('CKD4_date', 'CVD_date'),
        'ckd04_to_death_months': ('CKD4_date', 'DEAD_date'),
        
        'ckd5a_to_ckd5b_months': ('CKD5A_date', 'CKD5B_date'),
        'ckd5a_to_cvd00_months': ('CKD5A_date', 'CVD_date'),
        'ckd5a_to_death_months': ('CKD5A_date', 'DEAD_date'),
        
        'ckd5b_to_cvd00_months': ('CKD5B_date', 'CVD_date'),
        'ckd5b_to_death_months': ('CKD5B_date', 'DEAD_date'),
        
        'cvd00_to_ckd3b_months': ('CVD_date', 'CKD3B_date'),
        'cvd00_to_ckd04_months': ('CVD_date', 'CKD4_date'),
        'cvd00_to_ckd5a_months': ('CVD_date', 'CKD5A_date'),
        'cvd00_to_ckd5b_months': ('CVD_date', 'CKD5B_date'),
        'cvd00_to_death_months': ('CVD_date', 'DEAD_date')
        }
    return transition_dict

def get_transition_code():
    transition_dict = {
        'ckd3a_to_ckd3b_months': 1,
        'ckd3a_to_ckd04_months': 2,
        'ckd3a_to_ckd5a_months': 3,
        'ckd3a_to_ckd5b_months': 4,
        'ckd3a_to_cvd00_months': 5,
        'ckd3a_to_death_months': 6,
        
        'ckd3b_to_ckd04_months': 7,
        'ckd3b_to_ckd5a_months': 8,
        'ckd3b_to_ckd5b_months': 9,
        'ckd3b_to_cvd00_months': 10,
        'ckd3b_to_death_months': 11,
        
        'ckd04_to_ckd5a_months': 12,
        'ckd04_to_ckd5b_months': 13,
        'ckd04_to_cvd00_months': 14,
        'ckd04_to_death_months': 15,
        
        'ckd5a_to_ckd5b_months': 16,
        'ckd5a_to_cvd00_months': 17,
        'ckd5a_to_death_months': 18,
        
        'ckd5b_to_cvd00_months': 19,
        'ckd5b_to_death_months': 20,
        
        'cvd00_to_ckd3b_months': 21,
        'cvd00_to_ckd04_months': 22,
        'cvd00_to_ckd5a_months': 23,
        'cvd00_to_ckd5b_months': 24,
        'cvd00_to_death_months': 25
        }
    return transition_dict
