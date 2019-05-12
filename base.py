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
    :param data
    :return: Boolean
    '''
    if isinstance(data, data_type):
        return True
    else:
        input_type=type(data)
        raise TypeError('Se should be %s instead of %s' % (data_type,input_type))

def check_dict(data):
    '''
    Check data is dict
    :param data: dict
    :return: Boolean
    '''
    return check_type(data,dict)

def check_dataframe(df):
    '''
    Check df is the pd.DataFrame
    :param df:DataFrame
    :return:Boolean
    '''
    return check_dataframe(df,pd.DataFrame)


def check_series(se):
    '''
    Check se is the pd.Series
    :param se:
    :return:Boolean
    '''
    return check_type(se,pd.Series)


def show_all_dataframe():
    pd.set_option('display.max_column',1000)
    pd.set_option('display.width',1000)




