import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from rimac_analytics_api.constants.constants import Constants
from rimac_analytics_api.common.core import *


def fill_zeros_dni(df, col_dni='DOCUMENTO', inplace=False):
    """
    Updates the column to have DNI format.

    Parameters
    ----------
    df: Dataframe.
    col_dni: str, default 'DOCUMENTO'.
        Column name of the DNI.
    inplace: bool, default False.
        Inplace in column.

    Returns
    -------
    (inplace) column series.
    """
    series = df[col_dni].map(lambda x: str(x).zfill(8))
    if inplace:
        df[col_dni] = series
    else:
        return series


def fill_missing_values(df, features, _id='ID_UNICO', inplace=False):
    """
    Fill missings of dataframe with backward and fordward method.

    Parameters
    ----------
    df: Dataframe.
    features: list
        Features to be filled.
    _id: Group By Variable.
    inplace: bool, default False.
        Inplace in column.

    Returns
    -------
    (inplace) Dataframe with filled missings.
    """
    if inplace:
        df2 = df
    else:
        df2 = df.copy()

    for feature in features:
        df2[feature] = df2.groupby(_id)[feature].bfill().ffill()
    return df2


def make_datetime(series, inplace=False, **kwargs):
    """
    Converts object series to a datetime series

    Parameters
    ----------
    series: pandas series.
        Pandas series to be converted.
    inplace: bool, default False.
        inplace the series?
    **kwargs: arguments of pd.to_datetime function.

    Returns
    -------
    pandas series.
    """
    dates = {date: pd.to_datetime(date, **kwargs) for date in series.unique()}
    dates = series.map(dates)
    if inplace:
        series = dates
    else:
        return dates


def rolling(df, operation, columns, groupby_column, window=0, min_periods=3):
    """
    Returns a DataFrame with the rolling sum of some columns.

    Parameters
    ----------
    df: Dataframe.
    operation: 'sum' or 'mean'.
        For rolling/expanding sum or mean respectively.
    columns: str or list.
        Column names to apply the rolling sum.
    groupby_column: str or list.
        Column names to use as a grouper.
    window: int, default 0.
        Window for the rolling. 0 is equivalent to a rolling historical sum/mean.
    min_periods: int, default 3.
        Minimum number of periods to show the rolling.

    Returns
    -------
    A Dataframe with the columns rolled.
    """
    if isinstance(columns, str):
        columns = [columns]
    if isinstance(groupby_column, str):
        groupby_column = [groupby_column]
    all_columns = columns + groupby_column

    if operation == 'sum':
        if window > 0:
            return \
                df[all_columns].groupby(groupby_column).rolling(
                    min_periods=min_periods, window=window).sum().reset_index(
                    drop=True)[columns]
        else:
            return \
                df[all_columns].groupby(groupby_column).expanding(
                    min_periods=min_periods).sum().reset_index(
                    drop=True)[columns]
    elif operation == 'mean':
        if window > 0:
            return \
                df[all_columns].groupby(groupby_column).rolling(
                    min_periods=min_periods, window=window).mean().reset_index(
                    drop=True)[columns]
        else:
            return \
                df[all_columns].groupby(groupby_column).expanding(
                    min_periods=min_periods).mean().reset_index(
                    drop=True)[columns]


def label_encoder(df, cols_cat, suffix='', inplace=False):
    """
    Transforms categorical columns to numerical.

    Parameters
    ----------
    df: Dataframe.
    cols_cat: list.
        List of columns to be encoded.
    suffix: str, default is ''.
        Suffix to be added for each column name.
    inplace: bool, default True.
        Inplace the dataframe.

    Returns
    -------
    if inplace return dictionary of label encodings,
    if not inplace return dataframe and dictionary.
    """
    df0 = df if inplace else df.copy()
    dict_le = {}
    for var in cols_cat:
        le = LabelEncoder()
        le.fit(list(df0[var].dropna()))
        df0.loc[df0[var].notnull(), var + suffix] = le.transform(df0[var].dropna())
        dict_le[var] = le
    if inplace:
        return dict_le
    else:
        return df0, dict_le


def keep_top_values(df, cols_cat, include=None, keep_na=True, suffix='', value_other='OTRO',
                    top_n_values=Constants.default_top_values, inplace=False):
    """
    Keep top 'n' most frequency values for each variable.

    Parameters
    ----------
    df: Dataframe.
    cols_cat: str, list.
        Column(s) to keep top 'n' values.
    include: list, dict, default None.
        list if is one column to transform, dict of format {columns(str): values_to_include(list)}.
    keep_na: bool, default True.
        Keep nan or fill them with the 'value_other' value.
    suffix:  str, default ''.
        suffix for the transformed columns, '' means same name.
    value_other: str, default 'OTRO'.
        New value for the non-top 'n' values for each variable.
    top_n_values: int, default is Constants.default_top_values.
        Top 'n' values for each column.
    inplace: bool, default False.
        Inplace in dataframe ?

    Returns
    -------
    (inplace) Dataframe.
    """
    if include is None:
        include = {}
    if isinstance(cols_cat, str):
        if isinstance(include, list):
            include = {cols_cat: include}
        cols_cat = [cols_cat]
    df0 = df if inplace else df[cols_cat].copy()
    for var in cols_cat:
        first = df0[var].value_counts().index[:top_n_values].tolist()
        if var in include.keys():
            first += include[var]
        if keep_na:
            first.append(np.nan)
        _idx = df0[var].isin(first)
        df0.loc[_idx, var + suffix] = df0[var]
        df0.loc[~_idx, var + suffix] = value_other
    if len(suffix) > 0: df0.drop(cols_cat, axis=1, inplace=True)
    if not inplace:
        return df0


def ordinal_to_numerical(df, cols_ord, suffix='_NUM', inplace=False):
    """
    Transform ordinal to numerical values.

    Parameters
    ----------
    df: Dataframe.
    cols_ord: list.
        List of columns to be encoded.
    suffix: str, default is '_NUM'.
        Suffix to be added for each column name.
    inplace: bool, default True.
        Inplace the dataframe.

    Returns
    -------
    (inplace) Dataframe.
    """
    df0 = df if inplace else df.copy()
    dict_le = label_encoder(df0, cols_cat=cols_ord, suffix=suffix, inplace=True)
    if inplace:
        return dict_le
    else:
        return df0, dict_le


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
        if np.issubdtype(col_type, np.number):
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int' or str(col_type)[:4] == 'uint':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
                    # else:
                    #    df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
