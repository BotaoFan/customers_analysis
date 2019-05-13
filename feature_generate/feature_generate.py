#-*- coding:utf-8 -*-
# @Time : 2019/5/10
# @Author : Botao Fan

import pandas as pd
import numpy as np
import sklearn.preprocessing as sk_preprocessing
from sklearn.preprocessing import OneHotEncoder

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
        Extract one column as single feature without any manipulate
        :param col_name:str
        :param feature_name:str
        :return:Series named feature_name
        '''
        data = self.data
        b.check_col_exist(data, col_name)
        feature = data[col_name].copy()
        if feature_name is None:
            feature_name = col_name
        feature.rename(feature_name)
        return feature

    def generate_log1p(self, col_name, feature_name = None):
        '''
        Extract one column as single feature and covert to ln(x)
        :param col_name: str
        :param feature_name: str
        :return: Series named feature_name
        '''
        feature = np.log1p(self.generate(col_name, feature_name))
        return feature


class FeatureOneHot(Feature):
    def __init__(self):
        pass

    def generate(self, se, col_name):
        b.check_series(se)
        ohe=OneHotEncoder()
        result_array = ohe.fit_transform(se.values.reshape(-1, 1)).toarray()
        se_values_count = len(se.unique())
        result_df = pd.DataFrame(result_array, index=se.index, columns=[ col_name+'_'+str(i) for i in range(se_values_count)])
        return result_df


#Single Feature
class FeatureAge(Feature):
    def __init__(self):
        pass

    def generate_from_num_birthday(self, birthday_se, compare_date=None):
        '''
        :param birthday_se: pandas.Series contains str or float like 19880329
        :param compare_date: datetime.datetime, calculate age from this date
        :return: Series named age contains float
        '''
        date_translator=dp_base.DateTranslator()
        birthday_se_date=date_translator.covert_str_date(birthday_se)
        return self.generate_from_date_birthday(birthday_se_date)

    def generate_from_date_birthday(self, birthday_se, compare_date=None):
        '''

        :param birthday_se: pandas.Series contains datetime
        :param compare_date: datetime.datetime, calculate age from this date
        :return:Series named age contains float
        '''
        b.check_series(birthday_se)
        if compare_date is None:
            compare_date = datetime.now()
        datetime_delta = compare_date-birthday_se
        age_se = datetime_delta.apply(lambda x: x.days/365.0)
        age_se.name = 'age'
        return age_se


class FeatureAgeBin(Feature):
    def generate_from_age(self, age_se, bins=None, right=False):
        '''
        Generate bined age form age
        :param age_se: Series which contain numeric
        :param bins: list
        :param right: if right is True then bins are (],otherwise are [)
        :return: Series
        '''
        b.check_series(age_se)
        if bins is None:
            bins = range(0, 110, 10)
        return pd.cut(age_se, bins, right=right)

    def generate_from_date(self, birthday_se, bins=None, right=False, compare_date=None):
        feat_age = FeatureAge()
        age_se=feat_age.generate_from_date_birthday(birthday_se, compare_date)
        return self.generate_from_age(age_se, bins, right)

    def generate_from_num(self, birthday_se, bins=None, right=False, compare_date=None):
        feat_age = FeatureAge()
        age_se = feat_age.generate_from_num_birthday(birthday_se, compare_date)
        return self.generate_from_age(age_se, bins, right)


class FeatsGroupTrade(Feature):
    def __init__(self,data, cust_id_col, date_col, aim_col, feat_name, window=5):
        self.cust_id_col = cust_id_col
        self.date_col = date_col
        self.aim_col = aim_col
        self.feat_name = feat_name

        date_array = data[date_col].unique().sort()
        date_len = (len(date_array)//window)*window
        date_array = date_array[len(date_array)-date_len:]
        date_dict = {'date': date_array, 'date_id': range(date_len), 'window_id': np.array(range(date_len)) // window}
        self.date_df = pd.DataFrame(date_dict)
        self.data = pd.merge(data[[cust_id_col, date_col, aim_col]], self.date_df,
                             how='inner', left_on=date_col, right_on='date')
        self.data_grouped=self.data.groupby([cust_id_col,'window_id'])[aim_col]

    def generate_window_count(self):
        '''
        Generate count of aim_col during every window
        :return: pandas.DataFrame
        '''
        data_window_count = self.data_grouped.count().unstack()
        col_names = data_window_count.columns
        col_names_new = {}
        feat_name = self.feat_name
        b.add_prefix_on_col_name(data_window_count,feat_name)
        return data_window_count

    def generate_whole_stats(self):
        data_whole_stats = self.data.groupby([self.cust_id_col])['sno'].describle()
        feat_name = self.feat_name
        b.add_prefix_on_col_name(data_whole_stats, feat_name)
        return data_whole_stats




















