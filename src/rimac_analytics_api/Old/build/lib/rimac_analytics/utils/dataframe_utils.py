import pandas as pd


class DataframeUtils(object):
    @classmethod
    def get_cuc_from_doc_in_df(cls, df, column_doctype, column_doc, column_cuc):
        df[column_cuc] = df[column_doctype].apply(str) + '-' + df[
            column_doc].apply(str)
        return df

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
    def get_numerical_columns(cls, df, columns_to_exclude):
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
