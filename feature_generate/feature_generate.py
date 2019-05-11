#-*- coding:utf-8 -*-
# @Time : 2019/5/10
# @Author : Botao Fan

import pandas as pd
import numpy as np
from datetime import datetime
from .. import base as b
from ..data_preprocess import base as dp_base


class Feature(object):
    def __init__(self):
        pass

    def generate(self):
        pass


# Base Feature
class FeatureExtractColumn(Feature):
    def __init__(self, data):
        '''
        :param data:DataFrame which index should be the index of feature DataFrame
        '''
        b.check_dataframe(data)
        self.data = data


    def generate(self,col_name,feature_name=None):
        '''
        :param col_name:str
        :param feature_name:str
        :return:Series named feature_name
        '''
        data = self.data
        b.check_col_exist(data, col_name)
        feature = data[col_name].copy()
        feature.rename(col_name)
        return feature



#Single Feature
class FeatureAge(Feature):
    def __init__(self):
        pass

    def generate_from_num_birthday(self, birthday_se, compare_date=None):
        '''
        :param birthbirthday_seday: pandas.Series contains str or float like 19880329
        :return: Series named age contains float
        '''
        date_translator=dp_base.DateTranslator()
        birthday_se_date=date_translator.covert_str_date(birthday_se)
        return self.generate_from_date_birthday(birthday_se_date)

    def generate_from_date_birthday(self, birthday_se, compare_date=None):
        '''

        :param birthday_se: pandas.Series contains datetime
        :param compare_date:
        :return:Series named age contains float
        '''
        b.check_series(birthday_se)
        if compare_date is None:
            compare_date = datetime.now()
        datetime_delta = compare_date-birthday_se
        age_se = datetime_delta.apply(lambda x: x.days/365.0)
        age_se.name = 'age'
        return age_se






