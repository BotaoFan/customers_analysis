#-*- coding:utf-8 -*-
# @Time : 2019/5/10
# @Author : Botao Fan

import pandas as pd
import numpy as np


def check_col_exist(df, col_name):
    '''
    Check is col_name is a name of df's columns
    :param df: DataFrame
    :param col_name: str
    :return: None
    '''
    if col_name in df.columns:
        return True
    else:
        raise KeyError("DataFrame doesn't contain column named %s" % col_name)


def check_type(data, data_type):
    '''
    Check if type of data is data_type
    :param data:
    :param data_type:
    :return: Boolean
    '''
    if isinstance(data, data_type):
        return True
    else:
        input_type = type(data)
        raise TypeError('Se should be %s instead of %s' % (data_type, input_type))


def check_dict(data):
    '''
    Check data is dict
    :param data: dict
    :return: Boolean
    '''
    return check_type(data, dict)


def check_dataframe(df):
    '''
    Check df is the pd.DataFrame
    :param df:DataFrame
    :return:Boolean
    '''
    return check_type(df, pd.DataFrame)


def check_series(se):
    '''
    Check se is the pd.Series
    :param se:
    :return:Boolean
    '''
    return check_type(se, pd.Series)


def show_all_dataframe():
    pd.set_option('display.max_column', 1000)
    pd.set_option('display.width', 1000)


def add_prefix_on_col_name(df, prefix=''):
    '''
    Add prefix for DataFrame
    :param df: DataFrame
    :param prefix: str
    :return: DataFrame
    '''
    check_dataframe(df)
    if len(prefix) > 0:
        prefix = prefix + '_'
    col_names = df.columns
    col_names_new = {}
    for c in col_names:
        col_names_new[c] = prefix + str(c)
    df.rename(columns=col_names_new,inplace=True)
    return df


def add_suffix_on_col_name(df, suffix=''):
    '''
    Add prefix for DataFrame
    :param df: DataFrame
    :param prefix: str
    :return: DataFrame
    '''
    check_dataframe(df)
    if len(suffix) > 0:
        suffix = '_' + suffix
    col_names = df.columns
    col_names_new = {}
    for c in col_names:
        col_names_new[c] = suffix + str(c)
    df.rename(columns=col_names_new,inplace=True)
    return df








