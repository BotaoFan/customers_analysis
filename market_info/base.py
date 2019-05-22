#-*- coding:utf-8 -*-
# @Time : 2019/5/22
# @Author : Botao Fan

import pandas as pd
import numpy as np
import tushare as ts
from .. import base as b


class Pro(object):
    '''
    Create and return tushare pro_api using token.
    '''
    def __init__(self, token='2ad42b0cc3eb93d83e6d1e4b3f39300968f888cd5810f44974a1cd83'):
        self.pro = ts.pro_api(token)

    def get_pro(self):
        return self.pro


class IndexInfo(object):
    def __init__(self):
        pro_generate = Pro()
        self.pro = pro_generate()

    def download_index_daily_val(self, ts_code, start_end='20170101', end_date='20181228'):
        '''
        Get daily index information by downloading with tushare.pro_api
        :param ts_code: str
        :param start_end: str
        :param end_date: str
        :return: pandas.DataFrame
        '''
        index_val = self.pro.index_daily(ts_code=ts_code, start_date=start_end, end_date=end_date)
        index_val['trade_date'] = pd.to_datetime(index_val['trade_date'], format='%Y%m%d', errors='coerce')
        return index_val

    def get_index_val_csv(self, src=''):
        '''
        Get daily index information from csv
        :param src: str
        :return: pandas.DataFrame
        '''
        index_val = pd.read_csv(src)
        index_val['trade_date'] = pd.to_datetime(index_val['trade_date'], format='%Y%m%d', errors='coerce')
        return index_val

    def get_index_daily_return(self, index_val):
        '''
        Calculate daily return and add "return" column in index_val
        :param index_val: pandas.DataFrame
        :return: None
        '''
        b.check_col_exist(index_val, 'close')
        index_val['return'] = index_val['close']/(index_val['close'].shift(1)+0.0)-1
        return None
