class Constants(object):

    default_packages = {
        'data_processing': [
            'import pandas as pd',
            'import numpy as np',
        ],
        'data_visualization': [
            'import matplotlib.pyplot as plt',
            'import seaborn as sns',
        ],
        'machine_learning': [
            'import lightgbm as lgb',
            'from sklearn import preprocessing',
            'from sklearn.metrics import (auc, accuracy_score, confusion_matrix, roc_auc_score, precision_score, recall_score)',
            'from sklearn.externals import joblib',
            'from pandas_ml import ConfusionMatrix',
        ],
        'others': [
            'import time',
            'import string',
            'from datetime import datetime',
            'from collections import Counter',
            'from IPython.display import Image',
        ]
    }

    object_variables = [
        'TDOC', 'TIPO_DOCUMENTO', 'TIPODOC',
        'DOCUMENTO', 'DOC', 'DOC_CONYUGE_FUENTE', 'NRO_DOCUMENTO', 'DOC_CONYUGE', 'DNI',
        'PLACA', 'PLACA_ANTERIOR',
        'ID_CORREDOR',
        'RUC',
    ]
    object_variables.extend([x.lower() for x in object_variables])

    dtype_dict = {x: 'object' for x in object_variables}
    na_values = ['-', 'ND', 'NA', 'NO DETERMINADO']
    na_values.extend([x.lower() for x in na_values])

    default_top_values = 6

    default_parameters = {
        'lgbm': {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': {'auc'},
            'num_leaves': 30,
            'learning_rate': 0.01,
            'max_depth': 8,
            'feature_fraction': 0.7,
            'bagging_freq': 1,
            'bagging_fraction': 0.9,
            'is_unbalance': True,
            'verbose': 1
        }
    }

    default_grid_parameters = {
        'lgbm': {
            'task': ['train'],
            'boosting_type': ['gbdt'],
            'objective': ['binary'],
            'metric': [{'auc'}],
            'num_leaves': [2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 16, 20, 25, 30],
            'learning_rate': [0.005, 0.01, 0.015, 0.02, 0.05, 0.1],
            'max_depth': [-1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 25, 50, 75, 100],
            'feature_fraction': [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],
            'bagging_freq': [1],
            'bagging_fraction': [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1],
            'reg_alpha': [1, 1.1, 1.2, 1.3, 1.4],
            'reg_lambda': [1, 1.1, 1.2, 1.3, 1.4],
            'is_unbalance': [True, False],
            'verbose': [0]
        }
    }

