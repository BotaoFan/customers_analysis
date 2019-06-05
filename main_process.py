#-*- coding:utf-8 -*-
# @Time : 2019/5/27
# @Author : Botao Fan
import pandas as pd
import numpy as np
import warnings
from datetime import datetime

from customers_analysis.data_preprocess import preprocess as pp
from customers_analysis.data_preprocess import base as dp_base
from customers_analysis import base as b

from customers_analysis.feature_generate import feature_generate as fg

warnings.filterwarnings('ignore')


if __name__ == '__main__':
    abs_path = '/Users/fan/PycharmProjects/machine_learning/customer_loss/'
    raw_data_path = abs_path + 'raw_data/raw_data_customers_loss_1701_1704/'

    trade_date, cust_info, cust_trade = pp.script_import_and_clean(raw_data_path, datetime(2017, 1, 1), datetime(2017, 3, 31), 5)

