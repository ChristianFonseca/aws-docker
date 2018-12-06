import os
import sys
import pandas as pd
from rimac_analytics_api.utils import (exploration as raex,
                                   feature_engineering as rafe,
                                   import_packages_string as raip,
                                   miscellaneous as rami,
                                   prospection as rapr)

from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer

import joblib
import s3io
import pickle
import numpy as np
import gc
import time
import datetime
from dateutil.relativedelta import relativedelta
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error


path_model = 'rimac-analytics-temporal/models/siniestralidad_veh/bases/final/'
path_common = 'rimac-analytics-temporal/common/bases/static/'


def impxgb(valores,variables):
    dictimp={variables[a]:valores[a] for a in range(0,len(variables)) }
    xgimp=sorted(dictimp.items(), key=lambda x: x[1],reverse=True)

    return xgimp

def filter_threshold(probabilities, threshold):
    return [1 if f >= threshold else 0 for f in probabilities]

def get_threshold_measures_df(probabilities, observed):
    steps = [x / 100.0 for x in range(0, 100, 1)]
    df = pd.DataFrame(columns=['Punto de corte', 'Recall', 'Accuracy', 'Precision'])

    for i in range(len(steps)):
        estimated_threshold = filter_threshold(probabilities, steps[i])
        row = [
            steps[i],
            recall_score(observed, estimated_threshold),
            accuracy_score(observed, estimated_threshold),
            precision_score(observed, estimated_threshold),
            #auc(observed, estimated_threshold),
            #auc(observed, estimated_threshold)*2 - 1
        ]
        df.loc[i] = row

    return df

def cols_tipos(df, exclude = [], cols_ord = [], Print = False):
    # Tipo de variable
    cols = [x for x in df.columns if x not in exclude]
    cols_cat = [x for x in list(df.select_dtypes(include=['object'])) if x not in exclude]
    cols_num = [x for x in list(df.select_dtypes(exclude=['object'])) if x not in exclude]

    # Categorías nominales y ordinales
    cols_nom = [x for x in cols_cat if x not in cols_ord]

    if Print:
        print ('Categóricas:\n', cols_cat)
        print ('\nCategóricas Ordinal:\n', cols_ord)
        print ('\nCategóricas Nominal:\n', cols_cat)
        print ('\nNuméricas:\n', cols_num)
    
    return cols, cols_cat, cols_num, cols_nom

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
    
def save_data(pc, obj, path_file, pkl = False, **kwargs):
    if pkl == True:
        bucket, key = path_file.split('/', maxsplit=1)
        if pc == 's3':
            with s3io.open('s3://{0}/{1}'.format(bucket, key), mode='w') as s3_file:
                joblib.dump(obj, s3_file)
        else:
            joblib.dump(obj, path_file)
    else:
        if pc == 's3':
            raex.toS3(obj, path_file, **kwargs)
        else:
            obj.to_csv(path_file, **kwargs)
    print(path_file + ' saved.')
    return

def read_data(pc, path_file, pkl = False, **kwargs):
    bucket, key = path_file.split('/', maxsplit=1)
    if pkl == True:
        if pc == 's3':
            with s3io.open('s3://{0}/{1}'.format(bucket, key), mode='r') as s3_file:
                obj = joblib.load(s3_file)
        else:
            obj = joblib.load(s3_file)
    else:
        s3_bool = False
        if pc == 's3':
            s3_bool = True
        obj = raex.readCSV(path_file, s3=s3_bool, print_info = False, **kwargs)
    return obj

def clean_vehicular_base(df):
    df['month_ini']=df['FECINICERT'].str[3:6]
    df['month_ini']=df['month_ini'].map({'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,'JUL':7,
        'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12})
    df['month_ini']=df['month_ini'].astype(int,errors='ignore')
    df['year_ini']=df['FECINICERT'].str[7:9].map(float)+2000
    df['PERINI']=(df['year_ini']*100+df['month_ini']).astype(int,errors='ignore')

    df['month_fin']=df['FECFINCERT'].str[3:6]
    df['month_fin']=df['month_fin'].map({'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,'JUL':7,
        'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12})
    df['month_fin']=df['month_fin'].astype(int,errors='ignore')
    df['year_fin']=df['FECFINCERT'].str[7:9].map(float)+2000
    df['PERFIN']=(df['year_fin']*100+df['month_fin']).astype(int,errors='ignore')

    df.loc[df['FECANUL']=='15-JUL-01','FECANUL']=np.nan
    df['month_anu_pol']=df['FECANUL'].str[3:6]
    df['month_anu_pol']=df['month_anu_pol'].map({'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,'JUL':7,
        'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12})
    df['month_anu_pol']=df['month_anu_pol'].astype(int,errors='ignore')
    df['year_anu_pol']=df['FECANUL'].str[7:9].astype(float,errors='ignore')
    df.loc[df['year_anu_pol'].notnull(),'year_anu_pol']=2000+df['year_anu_pol']
    df['PERANU_POL']=df['year_anu_pol']*100+df['month_anu_pol']

    df['month_anu_cert']=df['FECANUCERT'].str[3:6]
    df['month_anu_cert']=df['month_anu_cert'].map({'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,'JUL':7,
        'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12})
    df['month_anu_cert']=df['month_anu_cert'].astype(int,errors='ignore')
    df['year_anu_cert']=df['FECANUCERT'].str[7:9].astype(float,errors='ignore')
    df.loc[df['year_anu_cert'].notnull(),'year_anu_cert']=2000+df['year_anu_cert']
    df['PERANU_CERT']=df['year_anu_cert']*100+df['month_anu_cert']

    df['DURACION'] = 12
    df.loc[df['PERANU_CERT']<=df['PERANU_POL'],'DURACION'] = \
        (df['year_anu_cert']-df['year_ini'])*12+(df['month_anu_cert']-df['month_ini'])
    df.loc[df['PERANU_POL']<df['PERANU_CERT'],'DURACION'] = \
        (df['year_anu_pol']-df['year_ini'])*12+(df['month_anu_pol']-df['month_ini'])

    df = df[(df['year_fin']-df['year_ini']==1)&(df['month_fin']==df['month_ini'])]
    df = df[df['PRIMA_EMITIDA']>200]

    df.rename(columns={'MTO_SINIESTRO':'MTO_SINIESTRO_PAGADO','MTO_RESERVA':'MTO_SINIESTRO'},inplace=True)
    df['MTO_SINIESTRO'] = df['MTO_SINIESTRO'].fillna(0)

    return df

def target_formulation(df):
    df.sort_values(['CUC','PLACA','PERIODO'],ascending=True,inplace=True)
    for x in ['NRO_SINIESTROS', 'MTO_SINIESTRO']:
        df[x+'_1']=df.groupby(['CUC','PLACA'])[x].shift(1)
        df[x+'_2']=df.groupby(['CUC','PLACA'])[x].shift(2)
        df[x+'_3']=df.groupby(['CUC','PLACA'])[x].shift(3)
        df[x+'_4']=df.groupby(['CUC','PLACA'])[x].shift(4)
        df[x+'_5']=df.groupby(['CUC','PLACA'])[x].shift(5)
        df[x+'_6']=df.groupby(['CUC','PLACA'])[x].shift(6)
        df[x+'_7']=df.groupby(['CUC','PLACA'])[x].shift(7)
    print(' - Siniestros anteriores creados.')

    df['FLAG_RENOVACION']=0
    df.loc[df['NRO_SINIESTROS_1'].notnull(),'FLAG_RENOVACION']=1    

    df['TARGET1']=0
    df.loc[df['MTO_SINIESTRO']>4000,'TARGET1']=1
    print(' - TARGET 1.')

    df['TARGET2']=0
    df.loc[df['NRO_SINIESTROS']>0,'TARGET2']=1
    print(' - TARGET 2.')

    df['TARGET3']=df['MTO_SINIESTRO']
    print(' - TARGET 3.')

    return df

def feature_engineering(_df):
    nse_list = ['NSE_RIMAC_0','NSE_RIMAC_6','NSE_RIMAC_12','NSE_RIMAC_24','NSE_RIMAC']
    rcc_list = ['CAL_GRAL_0','CAL_GRAL_6','CAL_GRAL_12','CAL_GRAL_24']

    for x in nse_list: _df[x] = _df[x].map({'E':0, 
                             'D2':1,
                             'D1':2,
                             'C2':3,
                             'C1':4,
                             'B2':5,
                             'B1':6,
                             'A2':7,
                             'A1':8,
                             'A':9})
    print(' - NSE')
    
    for x in rcc_list: _df[x] = _df[x].map({'PER':0, 
                             'DUD':1,
                             'DEF':2,
                             'CPP':3,
                             'OK':4})
    print(' - RCC')

    _df['SEXO'] = _df['SEXO'].map({'F': 0,'M':1})
    print(' - SEXO')

    _df['SEGMENTO_INTERNO'] = _df['SEGMENTO_INTERNO'].map({'6.MASIVO':0,
                                                       '5.ORO 3':1,
                                                       '4.ORO 2':2,
                                                       '3.ORO 1':3,
                                                       '2.PLATINO 2':4,
                                                       '1.PLATINO 1':5})   
    print(' - SEGMENTO_INTERNO')

    _df.loc[_df['NSE_UBIGEO']=='P','NSE_UBIGEO']=np.nan
    _df['NSE_UBIGEO'] = _df['NSE_UBIGEO'].map({'E':0,
                                           'D':1,
                                           'C':2,
                                           'B':3,
                                           'A':4,})
    print(' - NSE_UBIGEO')

    _df['GRADO_INSTRUCCION2'] = _df['GRADO_INSTRUCCION'].str[-3:]
    _df['GRADO_INSTRUCCION'] = _df['GRADO_INSTRUCCION'].str[0:3]
    _df['GRADO_INSTRUCCION'] = _df['GRADO_INSTRUCCION'].map({'EDU':0,
                                                         'ILE':0,
                                                         'PRI':0,
                                                         'SEC':2,
                                                         'TEC':4,
                                                         'SUP':6})
    _df.loc[(_df['GRADO_INSTRUCCION']==0)&(_df['GRADO_INSTRUCCION2']=='ETA'),'GRADO_INSTRUCCION']=1
    _df.loc[(_df['GRADO_INSTRUCCION']==2)&(_df['GRADO_INSTRUCCION2']=='ETA'),'GRADO_INSTRUCCION']=3
    _df.loc[(_df['GRADO_INSTRUCCION']==4)&(_df['GRADO_INSTRUCCION2']=='ETA'),'GRADO_INSTRUCCION']=5
    _df.loc[(_df['GRADO_INSTRUCCION']==6)&(_df['GRADO_INSTRUCCION2']=='ETA'),'GRADO_INSTRUCCION']=7
    print(' - GRADO_INSTRUCCION')

    _df.drop(columns=['GRADO_INSTRUCCION2'],inplace=True)
    _df['EDAD'] = 2018-_df['FEC_NACIMIENTO'].str[-4:].map(float)
    print(' - EDAD')

    _df['DEP_PER'] = _df.loc[_df['UBIGEO']>100000,'UBIGEO'].map(str).str[0:4]
    _df.loc[_df['UBIGEO']<100000,'DEP_PER'] = '0' + _df['UBIGEO'].map(str).str[0:3]

    _df['DEP_EMP'] = _df.loc[_df['UBIGEO_EMP']>100000,'UBIGEO_EMP'].map(str).str[0:4]
    _df.loc[_df['UBIGEO_EMP']<100000,'DEP_EMP'] = '0' + _df['UBIGEO_EMP'].map(str).str[0:3]
    print(' - UBIGEO')

    _df['SALDO_VEH_SBS_null']=_df[['SALDO_VEH_SBS_0','SALDO_VEH_SBS_6','SALDO_VEH_SBS_12','SALDO_VEH_SBS_24']].isnull().sum(axis=1)

    num_cols_list = ['CAL_OK_','CAL_CPP_','CAL_DEF_','CAL_DUD_','CAL_PER_',
                  'CLI_BANCOS_','CLI_FINANCIERAS_','CLI_CMAC_','CLI_CRAC_','CLI_EDPYMES_',   
                  'HIP_INI_','AU_INI_','NUM_ENT_SBS_','NUM_TC_SBS_','NUM_VEHIC_SBS_',
                  'NUM_HIPOT_SBS_','NUM_PP_SBS_','NUM_MICRO_SBS_','NUM_PEQUENA_SBS_',
                  'SALDO_SBS_','LINEA_TCMAX_','SALDO_TC_SBS_','SALDO_VEH_SBS_',
                  'SALDO_HIP_SBS_','SALDO_PP_SBS_','SALDO_MICRO_SBS_','SALDO_PEQUENA_SBS_',
                  'INGRESO_','INGRESO_RUC_PRINCIPAL_']
    
    for feature in num_cols_list:    
        _df['{}0'.format(feature)] = _df['{}0'.format(feature)].fillna(0)
        _df['{}6'.format(feature)] = _df['{}6'.format(feature)].fillna(0)
        _df['{}12'.format(feature)] = _df['{}12'.format(feature)].fillna(0)
        _df['{}24'.format(feature)] = _df['{}24'.format(feature)].fillna(0)
        _df['{}D1'.format(feature)] = _df['{}6'.format(feature)] - _df['{}0'.format(feature)]
        _df['{}D2'.format(feature)] = _df['{}12'.format(feature)] - _df['{}0'.format(feature)]
        _df['{}D3'.format(feature)] = _df['{}24'.format(feature)] - _df['{}0'.format(feature)]
        _df=_df.drop(columns=['{}6'.format(feature),'{}12'.format(feature),'{}24'.format(feature)])  
    print(' - CALIFICACIONES, SALDOS, LINEA, INGRESOS')

    _df.loc[_df['PUERTAS']==' ','PUERTAS']=np.nan
    _df['PUERTAS'] = _df['PUERTAS'].fillna(np.nan).map(float)
    _df.loc[_df['ASIENTOS']==' ','ASIENTOS']=np.nan
    _df['ASIENTOS'] = _df['ASIENTOS'].fillna(np.nan).map(float)
    _df['AUTO_DIFF'] = _df['VALOR_AUTO_NUEVO'] - _df['SUMASEG']
    print(' - CARACTERISTICAS DEL AUTOMOVIL')

    _df['NRO_RENOVACIONES'] = _df[['NRO_SINIESTROS_1','NRO_SINIESTROS_2','NRO_SINIESTROS_3','NRO_SINIESTROS_4',
                                  'NRO_SINIESTROS_5','NRO_SINIESTROS_6','NRO_SINIESTROS_7']].notnull().sum(axis=1) 

    _df['NRO_SINIESTROS_TOTAL'] = _df[['NRO_SINIESTROS_1','NRO_SINIESTROS_2','NRO_SINIESTROS_3','NRO_SINIESTROS_4',
                                  'NRO_SINIESTROS_5','NRO_SINIESTROS_6','NRO_SINIESTROS_7']].sum(axis=1) 
    _df['NRO_SINIESTROS_PROMEDIO'] = _df['NRO_SINIESTROS_TOTAL']/_df['NRO_RENOVACIONES']
    _df['MTO_SINIESTRO_TOTAL'] = _df[['MTO_SINIESTRO_1','MTO_SINIESTRO_2','MTO_SINIESTRO_3','MTO_SINIESTRO_4',
                                  'MTO_SINIESTRO_5','MTO_SINIESTRO_6','MTO_SINIESTRO_7']].sum(axis=1) 
    _df['MTO_SINIESTRO_PROMEDIO'] = _df['MTO_SINIESTRO_TOTAL']/_df['NRO_RENOVACIONES']

    _df['RECENCIA'] = 0
    _df.loc[_df['NRO_SINIESTROS_1']>=1,'RECENCIA'] = 1
    _df.loc[(_df['NRO_SINIESTROS_1']==0)&(_df['NRO_SINIESTROS_2']>=1),'RECENCIA'] = 2
    _df.loc[(_df['NRO_SINIESTROS_1']==0)&(_df['NRO_SINIESTROS_2']==0)&(_df['NRO_SINIESTROS_3']>=1),'RECENCIA'] = 3
    _df.loc[(_df['NRO_SINIESTROS_1']==0)&(_df['NRO_SINIESTROS_2']==0)&(_df['NRO_SINIESTROS_3']==0)
          &(_df['NRO_SINIESTROS_4']>=1),'RECENCIA'] = 4
    _df.loc[(_df['NRO_SINIESTROS_1']==0)&(_df['NRO_SINIESTROS_2']==0)&(_df['NRO_SINIESTROS_3']==0)
          &(_df['NRO_SINIESTROS_4']==0)&(_df['NRO_SINIESTROS_5']>=1),'RECENCIA'] = 5
    _df.loc[(_df['NRO_SINIESTROS_1']==0)&(_df['NRO_SINIESTROS_2']==0)&(_df['NRO_SINIESTROS_3']==0)
          &(_df['NRO_SINIESTROS_4']==0)&(_df['NRO_SINIESTROS_5']==0)&(_df['NRO_SINIESTROS_6']>=1),'RECENCIA'] = 6
    _df.loc[(_df['NRO_SINIESTROS_1']==0)&(_df['NRO_SINIESTROS_2']==0)&(_df['NRO_SINIESTROS_3']==0)
          &(_df['NRO_SINIESTROS_4']==0)&(_df['NRO_SINIESTROS_5']==0)&(_df['NRO_SINIESTROS_6']==0)
          &(_df['NRO_SINIESTROS_7']>=1),'RECENCIA'] = 7
    
    print(' - MONTO Y NUMERO DE SINIESTROS')
    
    text_var = ['DESCRIP','REVISDES','COMBO_PRODUCTOS']
    for var in text_var:
        _df[var]=_df[var].str.replace("(","")
        _df[var]=_df[var].str.replace(")","")
        _df[var]=_df[var].str.replace("/"," ")
        _df[var]=_df[var].str.replace("-","")  
    
        a=CountVectorizer(ngram_range=(1,1),min_df=1000)
        b=a.fit_transform(_df[var].dropna())
        for x in a.get_feature_names():
            _df[x]=np.nan
        _df.loc[_df[var].notnull(),a.get_feature_names()]=b.todense()
        _df.drop(columns=var,inplace=True)

    print(' - COUNT_VECTORIZER')
    
    return _df










def main(get_base_vehicular    = 'BASE_VEHICULAR_FINAL_2', 
         get_data_renovaciones = 'renovaciones_enero',
         get_parque_vehicular  = 'parque_vehicular_201801'):
    
    base_vehicular    = get_base_vehicular
    data_renovaciones = get_data_renovaciones
    parque_vehicular  = get_parque_vehicular
    
    ################################################################
    # CREACION DE BASE DE ENTRENAMIENTO
    ################################################################
    print('========================================================')
    print('CREACION DE BASE DE ENTRENAMIENTO')
    print('========================================================')
    ts = time.time()
    periodo_base = datetime.datetime.fromtimestamp(ts).strftime('%Y%m')
        
    output_name_training_base = f'training_base_{periodo_base}'

    print('Cargando Bases...',end='-')
    df = raex.readCSV(path_model+f'{base_vehicular}.csv.gz',s3=True,print_info=True)
    df_veh = raex.readCSV(path_common+f'{parque_vehicular}.csv.gz',s3=True,print_info=True,
                     usecols=['PLACA','MOTOR','MARCA_REVISADA','MODELO','CLASE','LONGITUD','ANCHO','ALTURA',
                            'VALOR_AUTO_NUEVO','ANHO_FAB','COLOR1','PUERTAS','ASIENTOS'])
    print('Terminado.')

    print('Limpiando base vehicular...',end=' ')
    df = clean_vehicular_base(df)
    print('Terminado..')

    print('Anhadiendo parque vehicular...',end=' ')
    df_1 = df[df['PLACA'].notnull()]
    df_2 = df[df['PLACA'].isnull()]
    df_1 = pd.merge(df_1,df_veh,on='PLACA',how='left')
    df_2 = pd.merge(df_2,df_veh,left_on='NRO_MOTOR',right_on='MOTOR',how='left')
    df = df_1.append(df_2)

    df.drop_duplicates(['CUC','PLACA','PERIODO'],inplace=True)
    df = df.reset_index()
    del df_veh
    del df_1
    del df_2
    print('Terminado...')
    gc.collect()

    print('Creando el Target...')
    df = target_formulation(df)
    print('Terminado...')
    print('Feature Engineering...')
    df = feature_engineering(df)
    print('Terminado.')

    print('Label Encoding...')
    exclude = ['CONTRAT_CUC','ID_POLIZA','FECINIPOL','FECFINPOL','UBIGEO','UBIGEO_EMP','Unnamed: 0','Unnamed: 1',
            'FECANUL','TIT_CUC','ID_CERTIFIC','FECINICERT','FECFINCERT','FECANUCERT','PLACA_y','MTO_SINIESTRO_PAGADO',
            'AUTO','PLACA','NRO_MOTOR','PRIMA_CONTABLE','NRO_SINIESTROS','RUC_PRINCIPAL_0','Unnamed: 0.1',
            'MTO_SINIESTRO','month_ini','PERINI','month_fin','RUC_PRINCIPAL_6','index','VALOR_AUTO_NUEVO',
            'year_fin','PERFIN','month_anu_pol','year_anu_pol','PERANU_POL','ENTIDAD_PRINC_12','Unnamed: 0.1.1',
            'month_anu_cert','year_anu_cert','PERANU_CERT','DURACION','BANCO_PRINC_24','year_ini',
            'TARGET1','CUC','PERIODO','YEAR','COMBO_PRODUCTOS_2','FEC_NACIMIENTO','ENTIDAD_PRINC_24',
            'PRIMA_EMITIDA','BANCO_PRINC_6','ENTIDAD_PRINC_6','BANCO_PRINC_12','TARGET2','TARGET3',
            'RUC_PRINCIPAL_12','RUC_PRINCIPAL_24','PLACA_x','MOTOR','NRO_MOTOR','MOTOR_y','MOTOR_x']
            
    cols_ord = ['CONO_AGRUP','GIRO','TIPO_DOC_EMP','TIPO_EMPRESA',
                'BANCO_PRINC_0','ENTIDAD_PRINC_0','DEP_PER','DEP_EMP',
                'ESTADO_CIVIL','MODELO','CLASE','COLOR1','MARCA_REVISADA',
                'MODULAR','CLASEPLANVEH','TIPCANDES','CANALDES']

    cols, cols_cat, cols_num, cols_nom = cols_tipos(df, exclude,cols_ord, Print = False)

    index_categorical=[cols.index(x) for x in cols_ord]
    encoding = {}
    for l in cols_ord:
        df[l]=df[l].map(str)
        df[l]=df[l].fillna('NULL')
        df2=df[[l]]
        df2['copy']=df2[l]
        counts = pd.value_counts(df2['copy'])
        for val in counts[counts < 100].index:
            df2['copy'] = df2['copy'].replace(val,'otro')
        df2=df2.groupby(l).agg('first').reset_index()
        df=pd.merge(df,df2,on=l,how='left')
        df.loc[(df[l]!='NULL')&(df[l]!=df['copy']),l]='otro'
        df=df.drop(['copy'], axis=1) 
        le = preprocessing.LabelEncoder()
        le.fit(list(df[l]))
        df[l]=le.transform(df[l])
        encoding[l] = le
        print(f' - {l}')

    print('Terminado...')

    print('Guardando Data...')
    # Cols
    save_data('s3',cols,path_model+'v2_cols.pkl', pkl=True)
    # Categorical Cols
    save_data('s3',cols_ord,path_model+'v2_cols_ord.pkl', pkl=True)
    # Index Categorical
    save_data('s3',index_categorical,path_model+'v2_index_categorical.pkl', pkl=True)
    # Encoding
    save_data('s3',encoding,path_model+'v2_encoding.pkl', pkl=True)
    # Guardar Dataframe
    raex.toS3(df,path_model + f'{output_name_training_base}.csv.gz',compression='gzip')
    print('Terminado...')
    print('Los datos procesados fueron guardados en: ',path_model + f'{output_name_training_base}.csv.gz')

    ################################################################
    # ENTRENAMIENTO
    ################################################################
    print('========================================================')
    print('ENTRENAMIENTO')
    print('========================================================')

    df = raex.readCSV(path_model+f'{output_name_training_base}.csv.gz',s3=True, print_info=True)

    # Cols:
    cols = read_data('s3',path_model+'v2_cols.pkl',pkl = True)
    # Categorical Cols:
    cols_ord = read_data('s3',path_model+'v2_cols_ord.pkl',pkl = True)
    # Index Categorical:    
    index_categorical = read_data('s3',path_model+'v2_index_categorical.pkl',pkl = True)
    # Encoding:    
    encoding = read_data('s3',path_model+'v2_encoding.pkl',pkl = True)


    # Split Training and Testing
    maximum_date = datetime.datetime.strptime(str(max(df['PERINI'].unique())), "%Y%m")
    maximum_date_sub = maximum_date - relativedelta(months=18)

    maximum_date_train = int(maximum_date_sub.strftime("%Y%m"))
    date_test1_1 = int((maximum_date + relativedelta(months=1)).strftime("%Y%m"))
    date_test1_2 = int((maximum_date + relativedelta(months=2)).strftime("%Y%m"))
    
    date_test2_1 = int((maximum_date + relativedelta(months=3)).strftime("%Y%m"))
    date_test2_2 = int((maximum_date + relativedelta(months=4)).strftime("%Y%m"))
    
    date_test3_1 = int((maximum_date + relativedelta(months=5)).strftime("%Y%m"))
    date_test3_2 = int((maximum_date + relativedelta(months=6)).strftime("%Y%m"))
    

    train = df[(df['PERINI']>=201401)&(df['PERINI']<=maximum_date_train)]

    test1 = df[(df['PERINI']>=date_test1_1)&(df['PERINI']<=date_test1_2)]
    test2 = df[(df['PERINI']>=date_test2_1)&(df['PERINI']<=date_test2_2)]
    test3 = df[(df['PERINI']>=date_test3_1)&(df['PERINI']<=date_test3_2)]

    test_all = df[(df['PERINI']>=date_test1_1)&(df['PERINI']<=date_test3_2)]

    recent = df[(df['PERINI']>=date_test1_1)&(df['PERINI']<=int(maximum_date.strftime("%Y%m")))]

    test_list = ['test1','test2','test3','test_all','recent']

    
    # Modelamiento
    # GIRO
    cols = ['MODELO','MARCA_REVISADA', 'TIPCANDES',
            'COLOR1','SUMASEG','NRO_SINIESTROS_PROMEDIO',
            'EDAD','MTO_SINIESTRO_PROMEDIO',
            'PROB_F','LONGITUD','MTO_SINIESTRO_1','PRIMA_VIG_CLIENTE',
            'AUTO_DIFF','SALDO_TC_SBS_D1','SALDO_TC_SBS_D2','SALDO_TC_SBS_D3',
            'MTO_SINIESTRO_2','ANCHO','NRO_SINIESTROS_1','SALDO_SBS_D1',
            'SALDO_SBS_D3','ALTURA','SALDO_SBS_D2','LINEA_TCMAX_D3',
            'GRADO_INSTRUCCION','MTO_SINIESTRO_TOTAL','SALDO_SBS_0',
            'INGRESO_0','LINEA_TCMAX_0','INGRESO_RUC_PRINCIPAL_0','ANHO_FAB',
            'DEP_PER','LINEA_TCMAX_D2','SALDO_TC_SBS_0',
            'INGRESO_RUC_PRINCIPAL_D3','CANT_CERTIFICADOS_VIG','SALDO_PP_SBS_D3',
            'MTO_SINIESTRO_3','NSE_UBIGEO','INGRESO_D3','SALDO_PP_SBS_D1',
            'CAL_CPP_D2','SALDO_VEH_SBS_D2','INGRESO_RUC_PRINCIPAL_D2',
            'SALDO_VEH_SBS_D1','SALDO_PP_SBS_D2','INGRESO_D2','ASIENTOS',
            'INGRESO_RUC_PRINCIPAL_D1','SALDO_VEH_SBS_D3','NSE_RIMAC_6',
            'SALDO_PP_SBS_0','972','LINEA_TCMAX_D1','INGRESO_D1',
            'SALDO_VEH_SBS_0','CAL_GRAL_24','NSE_RIMAC_24','NUM_ENT_SBS_0',
            'NSE_RIMAC_12','NSE_RIMAC','CAL_GRAL_12','AU_INI_D1',
            'NUM_PRODUCTOS','CAL_OK_D3','06','NUM_TC_SBS_0','2013',
            'SEGMENTO_INTERNO','NRO_SINIESTROS_TOTAL','NUM_TC_SBS_D3',
            'AU_INI_D3','NUM_ENT_SBS_D3','SALDO_PEQUENA_SBS_D1','CLI_BANCOS_D3',
            'NSE_RIMAC_0','vehicular','CAL_GRAL_0','NUM_VEHIC_SBS_D1','marsh',
            'NUM_TC_SBS_D2','NRO_SINIESTROS_2','2015','AU_INI_D2','HIP_INI_0',
            'SEXO','CONO_AGRUP','culos','NUM_ENT_SBS_D2','satelital',
            'NRO_SINIESTROS_3','SALDO_HIP_SBS_0','tr','CAL_OK_D2','soat',
            'CAL_DEF_D2','usos','SALDO_HIP_SBS_D3','SALDO_VEH_SBS_null',
            'CAL_GRAL_6','CAL_CPP_D1','AU_INI_0','NUM_PP_SBS_D2','HIP_INI_D2',
            'SALDO_HIP_SBS_D2','CAL_OK_D1','HIP_INI_D3','NUM_RIESGOS','PUERTAS',
            'HIP_INI_D1','ENTIDAD_PRINC_0','MODULAR','15','08','SALDO_HIP_SBS_D1',
            '09','TIPO_EMPRESA','N_HIJOS','vehã','premier','NUM_TC_SBS_D1',
            'eps','pick','CAL_CPP_D3','NUM_ENT_SBS_D1'
        ]

    # Guardar Columnas
    save_data('s3',cols,path_model+'columns_used.pkl', pkl=True)

    # MODELO 1
    print('MODELO 1')
    X = train[cols].values
    y = train['TARGET1'].values.ravel()

    d1={}
    e={}
    mylist = list(range(1,6))
    for k in mylist:
        X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.25, random_state=k*10)

        train_set = lgb.Dataset(X_train, y_train)
        validation_sets = lgb.Dataset(X_validation, y_validation, reference=train_set)

        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': { 'auc'},
            'learning_rate': 0.01,
            'min_data_in_leaf': 100,
            'feature_fraction': 0.5,
            'bagging_freq':1,
            'bagging_fraction': 0.8,
            'verbose': 1,
            'is_unbalanced':True
        }
    
        start_time = time.clock()
        model = lgb.train(
            params,
            train_set,
            num_boost_round=5000,
            valid_sets=validation_sets,
            early_stopping_rounds=100,
            categorical_feature=index_categorical,
            verbose_eval=2,
            )
        
        d1[k]=model
        test=model.predict(X_validation, num_iteration=model.best_iteration)
        e[k]=roc_auc_score(y_validation, test)

    # Guardar datos
    save_data('s3',d1,path_model+'models_1.pkl', pkl=True)
    save_data('s3',e,path_model+'auc_1.pkl', pkl=True)

    # MODELO 2
    print('MODELO 2')
    X = train[cols].values
    y = train['TARGET2'].values.ravel()

    d2={}
    e={}
    mylist = list(range(1,6))
    for k in mylist:
        X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.25, random_state=k*10)

        train_set = lgb.Dataset(X_train, y_train)
        validation_sets = lgb.Dataset(X_validation, y_validation, reference=train_set)

        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': { 'auc'},
            'learning_rate': 0.01,
            'min_data_in_leaf': 30,
            'feature_fraction': 0.5,
            'bagging_freq':1,
            'bagging_fraction': 0.8,
            'verbose': 1,
            'is_unbalanced':True
        }
    
        start_time = time.clock()
        model = lgb.train(
            params,
            train_set,
            num_boost_round=5000,
            valid_sets=validation_sets,
            early_stopping_rounds=100,
            categorical_feature=index_categorical,
            verbose_eval=2,
            )
        
        d2[k]=model
        test=model.predict(X_validation, num_iteration=model.best_iteration)
        e[k]=roc_auc_score(y_validation, test)

    # Guardar datos
    save_data('s3',d2,path_model+'models_2.pkl', pkl=True)
    save_data('s3',e,path_model+'auc_2.pkl', pkl=True)

    # MODELO 3 / df2
    print('MODELO 3')
    X = train[train['MTO_SINIESTRO']>0][cols].values
    y = train[train['MTO_SINIESTRO']>0]['TARGET3'].values.ravel()

    d3={}
    e={}
    mylist = list(range(1,6))
    for k in mylist:
        X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.25, random_state=k*10)

        train_set = lgb.Dataset(X_train, y_train)
        validation_sets = lgb.Dataset(X_validation, y_validation, reference=train_set)

        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': { 'l2'},
            'learning_rate': 0.01,
            'min_data_in_leaf': 30,
            'feature_fraction': 0.5,
            'bagging_freq':1,
            'bagging_fraction': 0.8,
            'verbose': 1,
            'is_unbalanced':True
        }
    
        start_time = time.clock()
        model = lgb.train(
            params,
            train_set,
            num_boost_round=5000,
            valid_sets=validation_sets,
            early_stopping_rounds=100,
            categorical_feature=index_categorical,
            verbose_eval=2,
            )
        
        d3[k]=model
        test=model.predict(X_validation, num_iteration=model.best_iteration)
        e[k]=mean_squared_error(y_validation, test)**0.5

    # Guardar datos
    print('Guardando Modelos...',end=' ')
    save_data('s3',d3,path_model+'models_3.pkl', pkl=True)
    save_data('s3',e,path_model+'auc_3.pkl', pkl=True)
    print('Terminado.')

    ################################################################
    # PROSPECCION
    ################################################################
    print('========================================================')
    print('PROSPECCION')
    print('========================================================')

    output_prospection_name = 'ren'

    # Modelo 1:
    d1 = read_data('s3',path_model+'models_1.pkl',pkl = True)
    # Modelo 2:
    d2 = read_data('s3',path_model+'models_2.pkl',pkl = True)
    # Modelo 3:    
    d3 = read_data('s3',path_model+'models_3.pkl',pkl = True)

    # Columns
    cols = read_data('s3',path_model+'columns_used.pkl',pkl = True)
    
    print('Cargando bases de renovaciones y personas...')
    df_ren = raex.readCSV(path_model + f'{data_renovaciones}.csv',s3=True, print_info=True)
    df_ase = raex.readCSV(path_common+'base_id_persona.csv.gz',s3=True, print_info=True)

    df_ren = pd.merge(df_ren,df_ase,left_on='ID_ASEGURADO ',right_on='ID_PERSONA',how='inner')
    df_ren = df_ren[['CUC','PLACA','ID_ASEGURADO ']]
    df_ren.rename(columns={'ID_ASEGURADO ':'ID_ASEGURADO'},inplace=True)
    df_ren['ren_veh'] = 1
    print('Terminado.')

    print('Creando montos y numero de siniestros anteriores...', end=' ')
    df2 = df.copy()
    df = df2

    df.sort_values(['CUC','PLACA','PERIODO'],ascending=True,inplace=True)
    for x in ['NRO_SINIESTROS', 'MTO_SINIESTRO']:
        df[x+'_1']=df.groupby(['CUC','PLACA'])[x].shift(0)
        df[x+'_2']=df.groupby(['CUC','PLACA'])[x].shift(1)
        df[x+'_3']=df.groupby(['CUC','PLACA'])[x].shift(2)
        df[x+'_4']=df.groupby(['CUC','PLACA'])[x].shift(3)
        df[x+'_5']=df.groupby(['CUC','PLACA'])[x].shift(4)
        df[x+'_6']=df.groupby(['CUC','PLACA'])[x].shift(5)
        df[x+'_7']=df.groupby(['CUC','PLACA'])[x].shift(6)
    print('Terminado.')
        
    df['FLAG_RENOVACION']=0
    df.loc[df['NRO_SINIESTROS'].notnull(),'FLAG_RENOVACION']=1    

    df = df[df['PERIODO']>=201706]
    df = pd.merge(df,df_ren,on=['CUC','PLACA'],how='left')
    df = df[df['ren_veh']==1]

    print('Desarrollando prospecciones...',end=' ')
    mylist = list(range(1,6))
    pros = df
    p=[]
    for k in mylist:
        pros['prob2_{}'.format(k)]=d2[k].predict(pros[cols], num_iteration=d2[k].best_iteration)
        p.append('prob2_{}'.format(k))
    pros['prob2'] = pros[p].mean(axis=1)

    p=[]
    for k in mylist:
        pros['prob3_{}'.format(k)]=d3[k].predict(pros[cols], num_iteration=d3[k].best_iteration)
        p.append('prob3_{}'.format(k))
    pros['prob3'] = pros[p].mean(axis=1)

    pros['PPR'] = pros['prob2']*pros['prob3']
    print('Terminado.')

    # Guardar Prospeccion
    print('Guardando prospecciones...')
    raex.toS3(pros[['PLACA','PPR']],path_model+f'{output_prospection_name}.csv',index=False)
    print(f'Guardado en {path_model}{output_prospection_name}.csv')

    print('\nPROCESO FINALIZADO')

if __name__ == "__main__":
    base_vehicular    = sys.argv[1]
    data_renovaciones = sys.argv[2]
    parque_vehicular  = sys.argv[3]

    main(base_vehicular, data_renovaciones, parque_vehicular)
