winSize = 30
samplingRate = 80
scalePath = 'StandardScalar_all_data.sav'
modelPath = 'LogisticRegression_all_data_F1score_0.70.sav'

timeString = 'HEADER_TIME_STAMP'

# feature_lis = ['X_DOMFREQ','Y_DOMFREQ','Z_DOMFREQ', 'X_DOMFREQ_POWER','Y_DOMFREQ_POWER','Z_DOMFREQ_POWER',
#                'X_TOTPOW','Y_TOTPOW','Z_TOTPOW', 'ORI_VAR_SUM',	'ORI_X_MEDIAN','ORI_Y_MEDIAN','ORI_Z_MEDIAN',
#                'SMV_ENERGY_SUM','SMV_ENERGY_VAR']

feature_lis = ['SMV_ENERGY_SUM','SMV_ENERGY_VAR', 'ORI_VAR_SUM', 'X_DOMFREQ','Y_DOMFREQ','Z_DOMFREQ','X_DOMFREQ_POWER',
               'Y_DOMFREQ_POWER','Z_DOMFREQ_POWER','X_TOTPOW','Y_TOTPOW','Z_TOTPOW']

MHEALTH_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S.%f"