import warnings


class Decorators(object):

    __warning_header = 'Decorator Warning: '
    __warning_messages = {
        'not_valid_column': 'Some columns names are not in the related '
                            'dataframe.',
        'invalid_types': 'Not valid types.'
    }

    @classmethod
    def __launch_warning(cls, warning_type):
        warnings.warn(
            cls.__warning_header +
            cls.__warning_messages[warning_type]
        )

    @classmethod
    def __validate_columns_list(cls, df, columns_list):
        for c in columns_list:
            if c not in df.columns.values:
                return False
        return True

    @classmethod
    def __get_param_from_arguments(cls, i, args, kwargs):
        if i < len(args):
            return args[i]
        else:
            kwargs_as_list = list(kwargs.items())
            return kwargs_as_list[i-len(args)][1]

    @classmethod
    def validate_columns(cls, df_arg_pos, column_arg_pos):
        def validate_column_generator(function_to_decorate):
            def inner_function(*args, **kwargs):
                df_to_validate = \
                    cls.__get_param_from_arguments(df_arg_pos, args, kwargs)
                columns_to_validate = \
                    cls.__get_param_from_arguments(column_arg_pos, args, kwargs)

                if isinstance(columns_to_validate, str):
                    if columns_to_validate in df_to_validate.columns.values:
                        return function_to_decorate(*args, **kwargs)
                    else:
                        cls.__launch_warning('not_valid_column')
                        return df_to_validate
                elif isinstance(columns_to_validate, list):
                    if cls.__validate_columns_list(df_to_validate,
                                                   columns_to_validate):
                        return function_to_decorate(*args, **kwargs)
                    else:
                        cls.__launch_warning('not_valid_column')
                        return df_to_validate
                else:
                    cls.__launch_warning('invalid_types')
                    return function_to_decorate(*args, **kwargs)

            return inner_function
        return validate_column_generator
