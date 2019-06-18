#-*- coding:utf-8 -*-
# @Time : 2019/5/26
# @Author : Botao Fan

import pandas as pd
import numpy as np

from .. import base as b
from . import base as dp_base
from ..feature_generate import feature_generate as fg
import datetime


def read_raw_data(path_file, path_columns=None, encoding='utf-8', dtype = None):
    if path_columns is None:
        result = pd.read_csv(path_file, encoding=encoding, header=True, dtype=dtype)
    else:
        result = pd.read_csv(path_file, encoding=encoding, header=None, dtype=dtype)
        col_names_df = pd.read_csv(path_columns, encoding=encoding, header=None)
        col_names_dict = dict()
        for i in range(col_names_df.shape[0]):
            col_names_dict[i] = col_names_df.loc[i, 1]
        result.rename(columns=col_names_dict, inplace=True)
    return result


def cut_window_day_id(date_index, window_days=5):
    date_index = pd.to_datetime(date_index)
    date_range = pd.DataFrame(range(len(date_index)), index=date_index, columns=['count'])
    date_range['window_id'] = date_range['count']//window_days
    max_num = len(date_index)//window_days*window_days
    date_range.loc[date_range['count'] >= max_num, 'window_id'] = np.nan
    date_range.drop(columns=['count'], inplace=True)
    return date_range



def script_import_and_clean(raw_data_path, start_date, end_date, window=5):
    #Load datea
    infos_path = raw_data_path + '../../infos/'
    trade_date = pd.read_csv(infos_path + 'trade_date.csv', encoding='utf-8')
    yyb_info = pd.read_csv(infos_path + 'yyb_area.csv', encoding='utf-8')

    trade_date['DateTime'] = pd.to_datetime(trade_date['DateTime'])
    trade_date_range = trade_date.loc[(trade_date['DateTime'] >= start_date) &
                                      (trade_date['DateTime'] <= end_date), 'DateTime']
    trade_date = cut_window_day_id(trade_date_range, window)
    cust_info = read_raw_data(raw_data_path + 'cust_info.csv', raw_data_path + 'cust_info_columns.csv')
    dtype_list = {17: str}
    cust_trade_kc_1 = read_raw_data(raw_data_path + 'cust_trade_kc_1.csv', raw_data_path + 'cust_trade_kc_columns.csv', dtype=dtype_list)
    cust_trade_kc_2 = read_raw_data(raw_data_path + 'cust_trade_kc_2.csv', raw_data_path + 'cust_trade_kc_columns.csv', dtype=dtype_list)
    cust_trade_kc_3 = read_raw_data(raw_data_path + 'cust_trade_kc_3.csv', raw_data_path + 'cust_trade_kc_columns.csv', dtype=dtype_list)
    cust_trade_kc_4 = read_raw_data(raw_data_path + 'cust_trade_kc_4.csv', raw_data_path + 'cust_trade_kc_columns.csv', dtype=dtype_list)
    cust_trade_kc_5 = read_raw_data(raw_data_path + 'cust_trade_kc_5.csv', raw_data_path + 'cust_trade_kc_columns.csv', dtype=dtype_list)
    cust_trade_rzq_1 = read_raw_data(raw_data_path + 'cust_trade_rzq_1.csv', raw_data_path + 'cust_trade_rzq_columns.csv', dtype=dtype_list)
    cust_trade_rzq_2 = read_raw_data(raw_data_path + 'cust_trade_rzq_2.csv', raw_data_path + 'cust_trade_rzq_columns.csv', dtype=dtype_list)
    cust_trade_rzq_3 = read_raw_data(raw_data_path + 'cust_trade_rzq_3.csv', raw_data_path + 'cust_trade_rzq_columns.csv', dtype=dtype_list)
    cust_trade = pd.concat([cust_trade_kc_1, cust_trade_kc_2, cust_trade_kc_3, cust_trade_kc_4, cust_trade_kc_5,
                          cust_trade_rzq_1, cust_trade_rzq_2, cust_trade_rzq_3], axis=0)
    del cust_trade_kc_1, cust_trade_kc_2, cust_trade_kc_3, cust_trade_kc_4, cust_trade_kc_5,\
        cust_trade_rzq_1, cust_trade_rzq_2, cust_trade_rzq_3

    # Translate date
    date_tran=dp_base.DateTranslator()
    cust_info['birthday_date'] = date_tran.covert_str_date(cust_info['birthday'])
    cust_info['khrq_date'] = date_tran.covert_str_date(cust_info['khrq'])
    cust_info['xhrq_date'] = date_tran.covert_str_date(cust_info['xhrq'])
    cust_trade['bizdate_date'] = date_tran.covert_str_date(cust_trade['bizdate'])
    # Generate y
    cust_info['y'] = 0
    cust_info.loc[cust_info['chg_rate'] <= -0.5, 'y'] = 1
    # Add information
    cust_info = pd.merge(cust_info, yyb_info, left_on='yyb', right_on='yyb', how='left')
    # Get window_day_id
    cust_trade = pd.merge(cust_trade, trade_date, left_on='bizdate_date', right_index=True, how='inner')

    #Deal with cust_trade
    cust_trade['buy'] = 1
    cust_trade.loc[cust_trade['fundeffect'] > 0, 'buy'] = 0
    cust_info.set_index('khh', inplace=True)
    return trade_date, cust_info, cust_trade








