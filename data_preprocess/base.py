#-*- coding:utf-8 -*-
# @Time : 2019/5/10
# @Author : Botao Fan

import pandas as pd
import numpy as np
from ..feature_generate import feature_generate as f


class DateTranslator(object):
    def __init__(self):
        self.data
        self.date_col_name
        #self.data_date=data[]

    def clean_str_date(df,col):
        # Translate all illegal date(in string format) to '18000101'
        df.loc[df[col].isnull(),col]='18000101'
        df.loc[df[col].astype(np.int32).astype(str).apply(lambda x: len(x) < 8), col]='18000101'


def show():
    f.show()
    print __name__
