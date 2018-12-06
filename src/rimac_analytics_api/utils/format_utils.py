from datetime import datetime
import pandas as pd


class FormatUtils(object):

    __str_date_format = '%d/%m/%Y'

    __dict_datetimes = {}
    __dict_dateints = {}

    @classmethod
    def get_year(cls, period):
        return period // 100

    @classmethod
    def get_month(cls, period):
        return period % 100

    @classmethod
    def gap_in_months_for_periods(cls, period_1, period_2):
        year_1 = cls.get_year(period_1)
        year_2 = cls.get_year(period_2)
        month_1 = cls.get_month(period_1)
        month_2 = cls.get_month(period_2)

        if year_1 == year_2:
            basic_difference = abs(month_1 - month_2) - 1
            if basic_difference < 0:
                basic_difference = 0
            return basic_difference
        elif year_1 > year_2:
            greater_year_dif = month_1
            smaller_year_dif = 12 - month_2
            basic_difference = greater_year_dif + smaller_year_dif
            additional_months_difference = (year_1 - year_2 - 1) * 12
            return basic_difference + additional_months_difference - 1
        elif year_1 < year_2:
            greater_year_dif = month_2
            smaller_year_dif = 12 - month_1
            basic_difference = greater_year_dif + smaller_year_dif
            additional_months_difference = (year_2 - year_1 - 1) * 12
            return basic_difference + additional_months_difference - 1

    @classmethod
    def get_difference_in_months(cls, datetime1, datetime2):
        difference = datetime1 - datetime2
        difference = abs(difference)
        return difference.days//30

    @classmethod
    def format_date_into_integer(cls, raw_date):
        if raw_date in cls.__dict_dateints:
            return cls.__dict_dateints[raw_date]
        else:
            datetime_obj = datetime.strptime(raw_date, cls.__str_date_format)
            if datetime_obj.month > 9:
                current_month = datetime_obj.month
            else:
                current_month = '0'+str(datetime_obj.month)

            formatted_date = '{year}{month}'.format(year=datetime_obj.year,
                                                    month=current_month)
            integer_date = int(formatted_date)
            cls.__dict_dateints[raw_date] = integer_date
            return integer_date

    @classmethod
    def get_datetime(cls, raw_date):
        if raw_date in cls.__dict_datetimes:
            return cls.__dict_datetimes[raw_date]
        else:
            new_datetime = datetime.strptime(raw_date, cls.__str_date_format)
            cls.__dict_datetimes[raw_date] = new_datetime
            return new_datetime

    @classmethod
    def format_series_string_into_datetime(cls, string_series, format=None,
                                           dayfirst=True):
        """
        Return a datetime series or string formated by a pattern.
        string_series: (str,series) Object to convert.
        format: (str pattern) Format of the object.
        dayfirst: Should the string_series have the day first ?
        """
        if isinstance(string_series, str):
            if len(string_series) == 6:
                string_series = str(string_series) + '01'
            return pd.to_datetime(string_series, format=format,
                                  dayfirst=dayfirst)
        else:
            if len(string_series[0]) == 6:
                string_series = string_series.map(lambda x: str(x) + '01')

            dates = \
                {date: pd.to_datetime(date, format=format, dayfirst=dayfirst)
                 for date in string_series.unique()}

            return string_series.map(dates)
