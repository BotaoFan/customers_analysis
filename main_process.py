#-*- coding:utf-8 -*-
# @Time : 2019/5/27
# @Author : Botao Fan
import pandas as pd
import numpy as np
from datetime import datetime

from data_preprocess import preprocess as pp
from feature_generate import feature_generate as fg

if __name__=='__main__':
    abs_path = '/Users/fan/PycharmProjects/machine_learning/customer_loss/'
    raw_data_path = abs_path + 'raw_data/raw_data_customers_loss_1701_1704/'
    yyb_area = pd.read_csv(abs_path + 'infos/yyb_area.csv', encoding='utf-8')

    trade_date, cust_info, cust_trade = pp.script_import_and_clean(raw_data_path, datetime(2017, 1, 1), datetime(2017, 3, 31), 5)
    #Add attrbute to cust_info,cust_trade
    cust_info = pd.merge(cust_info, yyb_area, left_on='yyb', right_on='yyb', how='left')
