import math

import pandas as pd
import numpy as np

def get_data_tuples(location):
    # Divide data in tuples as described in the paper.

    liver_train = pd.read_csv(f'{location}/liver_processed_train.csv')
    liver_test = pd.read_csv(f'{location}/liver_processed_test.csv')

    x_cols = ['AGE', 'ALBUMIN_TX', 'ASCITES_TX', 'BACT_PERIT_TCR', 'BMI_DON_CALC', 'CITIZENSHIP', 
    'CMV_STATUS', 'CREAT_TX', 'DAYSWAIT_CHRON', 'DGN2_TCR2', 'DGN_TCR1', 'DIAL_TX', 'EBV_SEROSTATUS', 
    'EDUCATION', 'ENCEPH_TX', 'ETHCAT', 'ETHNICITY', 'EVER_APPROVED', 'EXC_CASE','EXC_DIAG_ID','EXC_HCC',
    'FINAL_INR','FINAL_SERUM_SODIUM','FUNC_STAT_TCR','FUNC_STAT_TRR','GENDER','HBV_CORE','HBV_SUR_ANTIGEN',
    'HCC_DIAG','HCC_DIAGNOSIS_TCR','HCC_EVER_APPR','HCV_SEROSTATUS','HGT_CM_CALC','HGT_CM_TCR','HIV_SEROSTATUS',
    'INIT_ALBUMIN','INIT_ASCITES','INIT_BILIRUBIN','INIT_BMI_CALC','INIT_DIALYSIS_PRIOR_WEEK','INIT_ENCEPH' , 
    'INIT_INR','INIT_MELD_PELD_LAB_SCORE','INIT_SERUM_CREAT','INIT_SERUM_SODIUM','INIT_WGT_KG','INR_TX',
    'LIFE_SUP_TCR','LIFE_SUP_TRR','MALIG_TCR','MALIG_TRR','MED_COND_TRR','MELD_PELD_LAB_SCORE','NUM_PREV_TX',
    'ON_VENT_TRR','PORTAL_VEIN_TCR','PORTAL_VEIN_TRR','PREV_AB_SURG_TCR','PREV_AB_SURG_TRR','PREV_TX','TBILI_TX',
    'TIPSS_TCR','TIPSS_TRR','VENTILATOR_TCR','WGT_KG_CALC', 'WORK_INCOME_TCR','WORK_INCOME_TRR','abo','cancer',
    'diabbin','diag1', 'hosptime', 'insdiab', 'meldstat','meldstatinit','status1','statushcc',]

    x_cols = np.intersect1d(liver_train.columns.values, x_cols)

    o_cols = ['AGE_DON','ALCOHOL_HEAVY_DON','ANTIHYPE_DON','ARGININE_DON','BLOOD_INF_DON','BMI_DON_CALC',
    'BUN_DON','CARDARREST_NEURO','CDC_RISK_HIV_DON','CITIZENSHIP_DON','CLIN_INFECT_DON','CMV_DON',
    'COD_CAD_DON','CREAT_DON','DDAVP_DON','DIABETES_DON','EBV_IGG_CAD_DON','EBV_IGM_CAD_DON','ETHCAT_DON',
    'GENDER_DON','HBSAB_DON','HBV_CORE_DON','HEMATOCRIT_DON','HEPARIN_DON','HEP_C_ANTI_DON','HGT_CM_DON_CALC',
    'HISTORY_MI_DON','HIST_CANCER_DON','HIST_CIG_DON','HIST_COCAINE_DON','HIST_HYPERTENS_DON',
    'HIST_INSULIN_DEP_DON','HIST_OTH_DRUG_DON','INOTROP_SUPPORT_DON','INSULIN_DON','INTRACRANIAL_CANCER_DON',
    'NON_HRT_DON','PH_DON','PREV_TX_ANY','PROTEIN_URINE','PT_DIURETICS_DON','PT_STEROIDS_DON','PT_T3_DON',
    'PT_T4_DON','PULM_INF_DON','RECOV_OUT_US','RESUSCIT_DUR','SGOT_DON','SGPT_DON','SKIN_CANCER_DON',
    'TATTOOS','TBILI_DON','TRANSFUS_TERM_DON','URINE_INF_DON','VASODIL_DON','VDRL_DON','WGT_KG_DON_CALC',
    'abodon','coronary','death_mech_don_group','deathcirc','dontime','macro','micro']

    o_cols = np.intersect1d(liver_train.columns.values, o_cols)


    X_train = liver_train[x_cols]
    O_train = liver_train[o_cols]
    Y_train = liver_train[['PTIME']] 
    del_train = liver_train[['PSTATUS']] - 1 # PSTATUS is a censor variable

    X_test = liver_test[x_cols]
    O_test = liver_test[o_cols]
    Y_test = liver_test[['PTIME']]
    del_test = liver_test[['PSTATUS']] - 1

    return X_train, O_train, Y_train, del_train, X_test, O_test, Y_test, del_test