#-*- coding:utf-8 -*-
# @Time : 2019/5/10
# @Author : Botao Fan

import pandas as pd
import numpy as np
from datetime import datetime
from .. import base as b


class DateTranslator(object):
    '''
    Translate date col and return series col which has datetime type
    In above processing , translator doesn't change any data in original dataframe.
    '''
    def __init__(self):
        pass

    def clean_str_date(self, date_series):
        '''

        :param date_series
        :return:Series
        '''

        #nCheck
        b.check_series(date_series)
        # Translate all illegal date(in string format):null,length less then 8 to '18000101'
        se = date_series.copy()
        condition = (se.isnull()) | (se.astype(np.int32).astype(str).apply(lambda x: len(x) < 8))
        se.loc[condition, ] = '18000101'
        # se.loc[se[col_name].isnull(), ] = np.nan
        # se.loc[se[col_name].astype(np.int32).astype(str).apply(lambda x: len(x) < 8),]=np.nan
        return se

    def covert_str_date(self, date_series):
        se = date_series.copy()
        se = self.clean_str_date(se)
        se = pd.to_datetime(se.astype(np.int32).astype(str), format='%Y%m%d', errors='coerce')
        se[se == datetime(1800, 01, 01)] = np.nan
        return se


def value_map(se, map_dict):
    b.check_series(se)
    b.check_dict(map_dict)
    se=se.copy()
    for k in map_dict:
        v = map_dict[k]
        se[se == k] = v
    return se


