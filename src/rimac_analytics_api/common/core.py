from multiprocessing import cpu_count
from multiprocessing.dummy import Pool
import pandas as pd


def parallel_list(func, list, num_processes=None):
    if num_processes is None:
        num_processes = min(len(list), cpu_count())
    pool = Pool(num_processes)
    return pool.map(func, list)


def parallel_pandas(func, df, concat='columns', num_processes=None):
    """
    Apply a function separately to each column in a dataframe, in parallel.
    concat: for columns: like 'c%', for rows: like 'r%', for no concatenation: like 'n%'
    """
    concat = concat.lower()[0]
    axis = 1 if concat == 'c' else 0 if concat == 'r' else -1 if concat == 'n' else None
    _list = [df[col_name] for col_name in df.columns]
    results_list = parallel_list(func, _list, num_processes=num_processes)
    if axis == -1:
        return results_list
    else:
        return pd.concat(results_list, axis=axis)


def format_cols_param(cols, df):
    return df.columns if cols is None else [cols] if isinstance(cols, str) else cols


