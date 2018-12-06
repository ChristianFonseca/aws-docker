import pandas as pd
import boto3
from os import remove
from io import BytesIO, StringIO
from gzip import GzipFile
from rimac_analytics_api.constants.constants import Constants
from rimac_analytics_api.utils import miscellaneous as rami
from rimac_analytics_api.common.core import *


def print_basic_info(df):
    """
    Prints basic information: Nº of rows/columns and the column names.

    Parameters
    ----------
    df: Dataframe.

    Returns
    -------
    Prints info.
    """
    n_rows, n_cols = df.shape
    # print('(NRows: {:,} - NCols: {:,})'.format(n_rows, n_cols))
    print('{:<15}{:<15}'.format('NRows', 'NCols'))
    print('{:<15,}{:<15,}'.format(n_rows, n_cols))
    print(list(df.columns))


def readCSV(path_file, dtype='default', na_values='default', object_vars=[], print_info=True, s3=None, try_other_extension=True, **kwargs):
    """
    Read CSV from file. Default enconding 'latin1'.

    Parameters
    ----------
    path_file: str.
        Path (local or s3) of the csv file. If path begins with 's3://', s3 set to True.
    dtype: dict or 'default'.
        Default is Constants.dtype_dict
    na_values: list or 'default'.
        Default is Constants.na_values
    object_vars: list, default [].
        List of columns that should be consider as object(str).
    print_info: bool, default True.
        Prints basic information
    s3: bool, default None.
        If path_file is from s3 or local storage. If True of False use this instead of inferring from path.
    try_other_extension:
        Supported: 1) csv, 2) csv.gz
    **kwargs: kwargs.
        kwargs for the pd.read_csv function.

    Returns
    -------
    Dataframe.
    """

    if s3 and not path_file.startswith('s3://'):
        path_file = 's3://' + path_file
    if dtype == 'default':
        dtype = Constants.dtype_dict
    if na_values == 'default':
        na_values = Constants.na_values
    if 'encoding' not in kwargs:
        kwargs['encoding'] = 'latin1'
    for x in object_vars:
        dtype[x] = 'object'

    try:
        df = pd.read_csv(path_file, dtype=dtype, na_values=na_values, **kwargs)
    except:
        if path_file.endswith('.csv.gz'):
            df = pd.read_csv(path_file.replace('.csv.gz', '.csv'), dtype=dtype, na_values=na_values, **kwargs)
        elif path_file.endswith('.csv'):
            df = pd.read_csv(path_file.replace('.csv', '.csv.gz'), dtype=dtype, na_values=na_values, **kwargs)

    if print_info:
        print_basic_info(df)
        print()

    return df


def get_top_n_values(df, cols=None, top_n_values=Constants.default_top_values, parallel=True, num_processes=None):
    """
    Returns a DataFrame with the first 'n' values by each columns.

    Parameters
    ----------
    df: Dataframe.
    cols: (str, list), default None means all columns.
        Column(s) name for describing.
    top_n_values: int, default is Constants.default_top_values.
        Parameter for showing first 'n' values.
    parallel: bool, default True.
        Parallel proccess.
    num_processes: int, default None.
        Number of processes if parallel.

    Returns
    -------
    Dataframe of top_n_values for each column.
    """

    #Funcion for Parallel
    def func(series):
        firstn = series.value_counts().iloc[:top_n_values]
        firstn_index = firstn.index.tolist()
        firstn_N = firstn.values.tolist()
        for i in range(top_n_values - len(firstn)):
            firstn_index.append('')
            firstn_N.append(0)
        df_tmp = ['{} ({:,})'.format(x, y).replace(' (0)', '')
                     for x, y in zip(firstn_index, firstn_N)]
        return pd.DataFrame(data=df_tmp,
                            index=['TOP_' + str(x) for x in range(1, top_n_values + 1)],
                            columns=[series.name]).T

    if top_n_values > 0:
        cols = format_cols_param(cols, df)

        if parallel:
            return parallel_pandas(func, df[cols], concat='r', num_processes=num_processes)
        else:
            return pd.concat([func(df[col]) for col in cols])


def get_descriptive(df, cols=None, top_n_values=Constants.default_top_values, summary=True,
                    parallel=True, num_processes=None):
    """
    Returns a DataFrame with descriptives like dtype, N° Nulls, most frequency values, mean, etc for each column.

    Parameters
    ----------
    df: Dataframe.
    cols: (str, list), dafault None means all columns.
        Column(s) name for describing.
    top_n_values: int, default is Constants.default_top_values.
        Parameter for showing first 'n' values.
    summary: bool, default True.
        Show Descriptive statistics ?
    parallel: bool, default True.
        Parallel proccess.
    num_processes: int, default None.
        Number of processes if parallel.

    Returns
    -------
    Dataframe.
    """
    cols = format_cols_param(cols, df)
    df_copy = df[cols]
    df_types = df_copy.dtypes.to_frame()
    df_types.columns = ['dtype']
    df_types['dtype'] = df_types['dtype'].astype(str)

    df_types['Tipo'] = ''

    is_col_an_object = df_types['dtype'].isin(['object', 'category'])
    df_types['Tipo'][is_col_an_object] = 'Categoria'

    is_col_a_number = \
        [x.startswith(('int', 'float')) for x in df_types['dtype']]
    df_types['Tipo'][is_col_a_number] = 'Numero'

    nulls_quantity = df_copy.isnull().sum()
    nulls_percentage = nulls_quantity / float(df.shape[0])

    nulls_info_list = \
        ['{} ({:.1%})'.format(q, p) for q, p in zip(nulls_quantity,
                                                    nulls_percentage)]
    nulls_info_serie = pd.Series(nulls_info_list, index=df_copy.columns,
                                 name='N_Nulls')

    df_types = df_types.join(nulls_info_serie)\
                       .join(df_copy.nunique().to_frame('N_Unicos'))

    if top_n_values > 0:
        df_types = df_types.join(get_top_n_values(df_copy, cols, top_n_values,
                                                  parallel=parallel, num_processes=num_processes))
    if summary:
        df_types = df_types.join(df_copy.describe().T)
    return df_types


def get_columns_types(df, columns_to_exclude=[], ordinal_columns=[], flag_print=False):
    """
    Get column type from a Dataframe.

    Parameters
    ----------
    df: Dataframe.
    columns_to_exclude: list, default [].
        List of columns to exclude for the dtype analysis.
    ordinal_columns: list, default [].
        List of ordinal columns.
    flag_print: bool, default False.
        Print types.

    Returns
    -------
    dict.
    """
    valid_columns = [x for x in df.columns
                     if x not in columns_to_exclude]
    dtypes = df[valid_columns].dtypes.astype(str)

    categorical_columns = list(dtypes[dtypes.isin(['object', 'category'])].index)
    ordinal_columns = [x for x in categorical_columns
                       if x in ordinal_columns]
    nominal_columns = [x for x in categorical_columns
                       if x not in ordinal_columns]

    numerical_columns = list(dtypes[dtypes.str[:3].isin(['int', 'flo'])].index)
    other_columns = [x for x in valid_columns
                     if x not in categorical_columns and x not in numerical_columns]

    dic = {
        'valid': valid_columns,
        'categorical': categorical_columns,
        'nominal': nominal_columns,
        'ordinal': ordinal_columns,
        'numerical': numerical_columns,
        'other': other_columns,
        'exclude': columns_to_exclude
    }

    if flag_print:
        print('Categorical:\n', categorical_columns)
        print('\nCategorical Nominal:\n', nominal_columns)
        print('\nCategorical Ordinal:\n', ordinal_columns)
        print('\nNumerical:\n', numerical_columns)
        print('\nOther:\n', other_columns)
        print('\nExclude:\n', columns_to_exclude)

    return dic


def view_categorical_values(df, cols=None, exclude=[], max_length=200):
    """
    Print of values for each column.

    Parameters
    ----------
    df: Dataframe.
    cols: str, list, default None.
        (str) colum name of (list) list of column names.
    exclude: list, default [].
        list of columns to exclude.
    max_length: int or None, default 200.
        Max length of unique values to print. None means no limit.
    Returns
    -------
    Print.
    """
    cols = format_cols_param(cols, df)
    cols = [x for x in cols if x not in exclude]
    for var in cols:
        if max_length is None or df[var].nunique() < max_length:
            to_print = sorted(df[var].dropna().unique())
            print('{} ({:,}) \n{}\n {}\n'.format(var, len(to_print), '-'*(len(var)), to_print))


def toS3(df, path_file, low_memory=False, **kwargs):
    """
    Save dataframe to s3.

    Parameters
    ----------
    df: Dataframe.
    path_file: str.
        Path (local or s3) of the csv file.
    low_memory: bool, default False.
        If True, save to buffer (may increase ram) and store object to s3 path.
        If False, save to a temporal local file and move it to s3 path.
    **kwargs: kwargs.
        kwargs for the pd.to_csv function.
    Returns
    -------
    Print.    
    """
    if 'encoding' not in kwargs:
        kwargs['encoding'] = 'latin1'
    s3 = boto3.client('s3')
    bucket, file = rami.get_bucket_and_key_from_full_path(path_file)
    if low_memory: # Guardar en local (temporal), subir a S3, borrar el temporal
        temporal = '__temp__.csv'
        df.to_csv(temporal, **kwargs)
        execute = '{} s3://{}'.format(temporal, path_file)
        s3.upload_file(Filename=temporal, Bucket=bucket, Key=file)
        remove(temporal)
    else: # Guardar en objeto buffer (aumenta la memoria), subir a S3.
        csv_buffer = StringIO()
        compression = kwargs.pop('compression', None)
        df.to_csv(csv_buffer, **kwargs)
        
        if compression == 'gzip':
            csv_buffer.seek(0)
            gz_buffer = BytesIO()

            with GzipFile(mode='w', fileobj=gz_buffer) as gz_file:
                gz_file.write(bytes(csv_buffer.getvalue(), kwargs['encoding']))
            
            csv_buffer = gz_buffer
            del gz_buffer
        
        s3.put_object(Body=csv_buffer.getvalue(), Bucket=bucket, Key=file)







