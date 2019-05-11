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


def check_dataframe(df):
    '''
    Check df is the pd.DataFrame
    :param df:DataFrame
    :return:Boolean
    '''
    if isinstance(df, pd.DataFrame):
        return True
    else:
        input_type=type(df)
        raise TypeError('Df should be DataFrame instead of %s' % input_type)


def check_series(se):
    '''
    Check se is the pd.Series
    :param se:
    :return:Boolean
    '''
    if isinstance(se,pd.Series):
        return True
    else:
        input_type=type(se)
        raise TypeError('Se should be Series instead of %s' % input_type)


def show_all_dataframe():
    pd.set_option('display.max_column',1000)
    pd.set_option('display.width',1000)


