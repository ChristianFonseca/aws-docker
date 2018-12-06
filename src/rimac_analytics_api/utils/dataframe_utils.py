import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn import preprocessing
import matplotlib.pyplot as plt

class DataframeUtils(object):
    @classmethod
    def get_cuc_from_doc_in_df(cls, df, column_doctype, column_doc, column_cuc):
        df[column_cuc] = df[column_doctype].apply(str) + '-' + df[
            column_doc].apply(str)
        return df

    @classmethod
    def custom_function(cls):
        print('Hello World')

    @classmethod
    def print_basic_info(cls, df):
        print(len(df))
        print(len(df.columns))
        print(df.columns)

    @classmethod
    def dynamic_mean_by_n_months(cls, df, column_name, column_id, n):
        group_by_data = df.set_index(str(column_id), append=True).groupby(level=1)

        # resulting_series_with_mean = group_by_data[column_name].apply(
        #     pd.rolling_mean, n, 1).reset_index(str(column_id))
        #return resulting_series_with_mean[column_name]

        resulting_series_with_mean = group_by_data[column_name].apply(
            pd.rolling_mean, n, 1)
        return pd.Series(resulting_series_with_mean.values)

    @classmethod
    def get_categorical_columns(cls, df, columns_to_exclude):
        return \
            [x for x in list(df.select_dtypes(include=['object', 'category']))
             if x not in columns_to_exclude]

    @classmethod
    def get_numerical_columns(cls, df, columns_to_exclude=[]):
        return \
            [x for x in list(df.select_dtypes(include=['integer', 'floating']))
             if x not in columns_to_exclude]

    @classmethod
    def get_columns_by_types(cls, df, columns_to_exclude, ordinal_columns,
                             print_flag=False):
        valid_columns = [x for x in df.columns if x not in columns_to_exclude]
        categorical_columns = \
            cls.get_categorical_columns(df, columns_to_exclude)
        numerical_columns = \
            cls.get_numerical_columns(df, columns_to_exclude)
        other_columns = \
            [x for x in valid_columns if x not in categorical_columns
             and x not in numerical_columns]
        nominal_columns = \
            [x for x in categorical_columns if x not in ordinal_columns]

        if print_flag:
            print('Categóricas:\n', categorical_columns)
            print('\nCategóricas Ordinal:\n', ordinal_columns)
            print('\nCategóricas Nominal:\n', nominal_columns)
            print('\nNuméricas:\n', numerical_columns)
            print('\nOtros:\n', other_columns)
            print('\nExcluidos:\n', columns_to_exclude)

        return valid_columns, categorical_columns, \
               numerical_columns, nominal_columns

    @classmethod
    def get_top_n_values(cls, df, cols=None, n=6):
        """
        Returns a DataFrame with the first 'n' values by each columns.
        df: Dataframe.
        cols: (str, list) Column(s) name for describing.
        n: (int) Parameter for showing first 'n' values.
        """
        if cols is None:
            cols = df.columns
        elif isinstance(cols, str):
            cols = [cols]
        df_tmp = pd.DataFrame(columns=cols)
        for col in cols:
            firstn = df[col].value_counts().iloc[:n]
            firstn_index = firstn.index.tolist()
            firstn_N = firstn.values.tolist()
            for i in range(n - len(firstn)):
                firstn_index.append('')
                firstn_N.append(0)
            df_tmp[col] = ['{} ({:,})'.format(x, y) for x, y in
                           zip(firstn_index, firstn_N)]
            df_tmp = df_tmp.applymap(str)
            df_tmp = df_tmp.applymap(lambda x: x.replace('(0)', ''))
            df_tmp.index = ['V' + str(x) for x in range(1, n + 1)]
        return df_tmp.T

    @classmethod
    def get_descritives_from_dataframe(cls, df, cols=None, top_n_vals=6,
                                       summary=True):
        """
        Returns a DataFrame with descriptives like, dtype, N° Nulls,
        most frequency values, mean, etc by each column.
        df: Dataframe.
        cols: (str, list) Column(s) name for describing.
        n: (int) Parameter for showing first 'n' values.
        summary: (bool) Show Descriptive statistics ?
        """
        if cols is None:
            cols = df.columns
        df_copy = df[cols].copy()
        df_types = df_copy.dtypes.to_frame()
        df_types.columns = ['dtype']
        df_types['dtype'] = df_types['dtype'].astype(str)

        df_types['Tipo'] = ''

        is_col_an_object = df_types['dtype'] == 'object'
        df_types['Tipo'][is_col_an_object] = 'Categoria'

        is_col_a_number = \
            [x.startswith(('int', 'float')) for x in df_types['dtype']]
        df_types['Tipo'][is_col_a_number] = 'Numero'

        nulls_quantity = df_copy.isnull().sum()
        nulls_percentage = nulls_quantity / float(len(df_copy))

        nulls_info_list = \
            ['{} ({:.1%})'.format(q, p) for q, p in zip(nulls_quantity,
                                                        nulls_percentage)]
        nulls_info_serie = pd.Series(nulls_info_list, index=df_copy.columns,
                                     name='N_Nulls')

        df_types = df_types.join(nulls_info_serie)
        df_types = df_types.join(df_copy.nunique().to_frame('N_Unicos'))

        if top_n_vals > 0:
            df_types = df_types.join(cls.get_top_n_values(df_copy, cols,
                                                          top_n_vals))
        if summary:
            df_types = df_types.join(df_copy.describe().T)

        df_types.fillna('', inplace=True)
        return df_types

    @classmethod
    def rolling_sum(cls, df, columns, groupby_column, n=0, min_periods=3):
        """
        Returns a DataFrame with the rolling sum of some columns.
        df: Dataframe.
        columns: (str, list) Column names to apply the rolling sum.
        column_id: (str, list) Column names to use as a grouper.
        n: Window for the rolling. 0 is equivalent to a rolling historical sum.
        min_periods: Minimum number of periods to show the rolling.
        """
        if isinstance(columns, str):
            columns = [columns]
        if isinstance(groupby_column, str):
            groupby_column = [groupby_column]
        all_columns = columns + groupby_column

        if n > 0:
            return \
                df[all_columns].groupby(groupby_column).rolling(
                    min_periods=min_periods, window=n).sum().reset_index(
                    drop=True)[columns]
        else:
            return \
                df[all_columns].groupby(groupby_column).expanding(
                    min_periods=min_periods).sum().reset_index(
                    drop=True)[columns]

    @classmethod
    def rolling_mean(cls, df, columns, groupby_column, n=0, min_periods=3):
        """
        Returns a DataFrame with the rolling mean of some columns.
        df: Dataframe.
        columns: (str, list) Column names to apply the rolling mean.
        column_id: (str, list) Column names to use as a grouper.
        n: Window for the rolling. 0 is equivalent to a rolling historical mean.
        min_periods: Minimum number of periods to show the rolling.
        """
        if isinstance(columns, str):
            columns = [columns]
        if isinstance(groupby_column, str):
            groupby_column = [groupby_column]

        all_columns = columns + groupby_column
        if n > 0:
            return \
                df[all_columns].groupby(groupby_column).rolling(
                    min_periods=min_periods, window=n).mean().reset_index(
                    drop=True)[columns]
        else:
            return \
                df[all_columns].groupby(groupby_column).expanding(
                    min_periods=min_periods).mean().reset_index(
                    drop=True)[columns]

    @classmethod
    def view_categorical_values(cls, df, cols=None):
        cols = df.columns if cols is None else cols
        for var in df[cols].select_dtypes(['object']).columns:
            print(var, df[var].value_counts().sort_index().index, '\n')

    @classmethod
    def replace_values(cls, df, dic, inplace=False):
        cols = list(dic.keys())
        df0 = df if inplace else df.copy()
        for var, tranform in dic.items():
            df0[var] = df0[var].map(
                lambda x: tranform[x] if x in tranform.keys() else x)
        if not inplace:
            return df0

    @classmethod
    def categorical_to_numerical(cls, df, cols_cat, suffix='', inplace=False):
        df0 = df if inplace else df.copy()
        for var in cols_cat:
            le = preprocessing.LabelEncoder()
            le.fit(list(df0[var].dropna()))
            df0.loc[~df0[var].isnull(), var + suffix] = le.transform(
                df0[var].dropna())
        if not inplace:
            return df0

    @classmethod
    def keep_top_categorical_values(cls, df, cols=None, n=6):
        if cols is None:
            cols = df.columns
        df2 = df[cols].copy()
        for col in cols:
            df2[col + '2'] = 'OTRO'
            first = df2[col].value_counts().index[:n].tolist()
            df2[col + '2'][df2[col].isin(first)] = df2[col]
        df2.drop(cols, axis=1, inplace=True)
        return df2

    @classmethod
    def obtener_AUC_Gini_Roc2(cls, data, lgb, target, columns):
        df_p = data
        df_p['PREDS'] = lgb.predict(np.array(df_p[columns]),
                                    num_iteration=lgb.best_iteration)
        fpr, tpr, _ = metrics.roc_curve(df_p[target], df_p['PREDS'])
        roc_auc = metrics.auc(fpr, tpr)
        gini = 2*roc_auc - 1
        print('AUC: {}'.format(roc_auc))
        print('GINI: {}'.format(gini))
        plt.rcParams['figure.figsize'] = (10,10)
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC')
        plt.legend(loc="lower right")
        plt.show()
