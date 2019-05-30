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
        self.pro = pro_generate.get_pro()

    def download_index_daily_val(self, index_code, start_date='20150101', end_date='20181228'):
        '''
        Get daily index information by downloading with tushare.pro_api
        :param index_code: str
        :param start_end: str
        :param end_date: str
        :return: pandas.DataFrame
        '''
        index_val = self.pro.index_daily(ts_code=index_code, start_date=start_date, end_date=end_date)
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

    def get_index_attr(self,index_list, attr='close',start_date='20150101', end_date='20181228'):
        result_df = pd.DataFrame(index=pd.date_range(start=start_date, end=end_date))
        for idx in index_list:
            idx_val_df = self.download_index_daily_val(idx, start_date=start_date, end_date=end_date)
            result_df[idx] = idx_val_df.set_index('trade_date')[attr]
        return result_df



if __name__=="__main__":
    #Download important index close and daily return
    #上证综指：000001.SH, 深证成指：399001.SZ，上证50：000016.SH, 沪深300：000300.SH,中证500:000905.SH,中证1000:000852.SH,中小板指：399005.SZ
    index_list = ['000001.SH', '399001.SZ', '000016.SH', '000300.SH', '000905.SH', '000852.SH', '399005.SZ',
                  '801120.SI', '801750.SI', '801020.SI', '801030.SI', '801140.SI', '801150.SI', '801050.SI',
                  '801110.SI', '801210.SI', '801730.SI', '801890.SI', '801200.SI', '801710.SI', '801080.SI',
                  '801180.SI', '801130.SI', '801760.SI', '801880.SI', '801770.SI', '801160.SI', '801790.SI',
                  '801230.SI', '801720.SI', '801010.SI', '801740.SI', '801170.SI', '801780.SI', '801040.SI']
    start_date = '20150101'
    end_date = '20181228'
    indexInfo = IndexInfo()
    index_close = indexInfo.get_index_attr(index_list, 'close', start_date, end_date)
    index_daily_return = indexInfo.get_index_attr(index_list, 'pct_chg', start_date, end_date)

