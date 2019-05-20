#-*- coding:utf-8 -*-
# @Time : 2019/5/10
# @Author : Botao Fan

import pandas as pd
import numpy as np
import sklearn.preprocessing as sk_preprocessing
from sklearn.preprocessing import OneHotEncoder
from collections import defaultdict

from datetime import datetime
from .. import base as b
from ..data_preprocess import base as dp_base


def _check_func(func):
    str_func = {'count': len, 'sum': np.sum, 'max': np.max, 'min': np.min, 'median': np.median, 'mean': np.mean,
                'std': np.std}
    if isinstance(func, str):
        try:
            return str_func[func]
        except:
            print str_func
            raise KeyError('String function %s not in function list, try string above' % func)
    else:
        return func


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


class AbstractFeatGroupData(Feature):
    def __init__(self, data, cust_id_col, groupby_col, aim_col, func, feat_name):
        b.check_dataframe(data)
        b.check_col_exist(data, cust_id_col)
        b.check_col_exist(data, groupby_col)
        b.check_col_exist(data, aim_col)
        b.check_type(feat_name, str)
        self.cust_id = cust_id_col
        self.groupby_id = groupby_col
        self.aim_id = aim_col
        self.func = self._check_func(func)
        self.feat_name = feat_name
        self.data_grouped = data.groupby([cust_id_col, groupby_col])[aim_col]
        self.agg_data = None
        self.agg_data_norm = None
        self.agg_data_ratio = None


    def _check_func(self,func):
        str_func = {'count': len, 'sum': np.sum, 'max': np.max, 'min': np.min, 'median': np.median, 'mean': np.mean,
                    'std': np.std}
        if isinstance(func, str):
            try:
                return str_func[func]
            except:
                print str_func
                raise KeyError('String function %s not in function list, try string above' % func)
        else:
            return func

    def _generate_agg_data_polyfit(self, data, poly_n, prefix):
        reg_x = range(1, data.shape[1] + 1)
        polynomial = data.apply(lambda y: np.polyfit(reg_x, y.values, poly_n), axis=1)
        data_polyfit, data_polyfit_sign = pd.DataFrame(index=polynomial.index), pd.DataFrame(index=polynomial.index)
        for i in range(poly_n+1):
            data_polyfit[str(i)] = polynomial.apply(lambda x: x[poly_n-i-1])
            data_polyfit_sign[str(i)] = polynomial.apply(lambda x: 1 if (x[poly_n-i-1]) > 0 else 0)
        b.add_prefix_on_col_name(data_polyfit, self.feat_name + '_' + prefix + '_poly')
        b.add_prefix_on_col_name(data_polyfit_sign, self.feat_name + '_' + prefix + '_poly_sign')
        return data_polyfit, data_polyfit_sign

    def _generate_stats(self, data):
        data_stats = data.T.describe().T
        data_stats.drop(columns=['count'], inplace=True)
        data_stats['sum'] = data.sum(axis=1)
        data_stats['skew'] = data.skew(axis=1)
        data_stats['kurt'] = data.kurt(axis=1)
        b.add_prefix_on_col_name(data_stats, self.feat_name)
        return data_stats

    def generate_agg_data(self):
        func = self.func
        agg_data = self.data_grouped.agg(func).unstack()
        agg_data.fillna(0, inplace=True)
        b.add_prefix_on_col_name(agg_data, self.feat_name)
        self.agg_data = agg_data.copy()
        return agg_data

    def generate_agg_data_norm(self):
        agg_data = self.generate_agg_data() if self.agg_data is None else self.agg_data
        agg_data_norm = agg_data.sub(agg_data.mean(axis=1), axis='index').div(agg_data.std(axis=1), axis='index')
        b.add_prefix_on_col_name(agg_data_norm, 'norm')
        self.agg_data_norm = agg_data_norm.copy()
        return agg_data_norm

    def generate_agg_data_ratio(self):
        agg_data = self.generate_agg_data() if self.agg_data is None else self.agg_data
        agg_data_ratio = agg_data.div(agg_data.sum(axis=1), axis='index')
        b.add_prefix_on_col_name(agg_data_ratio, 'ratio')
        self.agg_data_ratio = agg_data_ratio.copy()
        return agg_data_ratio

    def generate_agg_data_ratio_polyfit(self, poly_n):
        agg_data_ratio = self.generate_agg_data_ratio() if self.agg_data_ratio is None else self.agg_data_ratio
        agg_data_ratio_polyfit, agg_data_ratio_polyfit_sign = \
            self._generate_agg_data_polyfit(agg_data_ratio, poly_n, 'ratio')
        return agg_data_ratio_polyfit, agg_data_ratio_polyfit_sign

    def generate_agg_data_norm_polyfit(self, poly_n):
        agg_data_norm = self.generate_agg_data_norm() if self.agg_data_norm is None else self.agg_data_norm
        agg_data_norm_polyfit, agg_data_norm_polyfit_sign = \
            self._generate_agg_data_polyfit(agg_data_norm, poly_n, 'norm')
        return agg_data_norm_polyfit, agg_data_norm_polyfit_sign

    def generate_data_stats(self):
        agg_data = self.generate_agg_data() if self.agg_data is None else self.agg_data
        agg_data_stats = self._generate_stats(agg_data)
        return agg_data_stats

    def generate_data_ratio_stats(self):
        agg_data_ratio = self.generate_agg_data_ratio() if self.agg_data_ratio is None else self.agg_data_ratio
        agg_data_ratio_stats = self._generate_stats(agg_data_ratio)
        return agg_data_ratio_stats

    def generate_data_norm_stats(self):
        agg_data_norm = self.generate_agg_data_norm() if self.agg_data_norm is None else self.agg_data_norm
        agg_data_norm_stats = self._generate_stats(agg_data_norm)
        return agg_data_norm_stats


class FeatureTradeTS(Feature):
    def __init__(self, data, cust_id_col, date_col, aim_col, func, feat_name, window=5):
        self.cust_id_col = cust_id_col
        self.date_col = date_col
        self.aim_col = aim_col
        self.feat_name = feat_name
        self.func = self._check_func(func)
        date_array = data[date_col].unique()
        date_array.sort()
        date_len = (len(date_array)//window)*window
        date_array = date_array[len(date_array)-date_len:]
        date_dict = {'date': date_array, 'date_id': range(date_len), 'window_id': np.array(range(date_len)) // window}
        self.window_num = date_len // window
        self.date_df = pd.DataFrame(date_dict)
        self.data = pd.merge(data[[cust_id_col, date_col, aim_col]], self.date_df,
                             how='inner', left_on=date_col, right_on='date')
        self.data_grouped = self.data.groupby([cust_id_col, 'window_id'])[aim_col]
        self.window_data = None
        self.window_data_ratio = None
        self.window_data_stats = None
        self.window_data_ratio_polyfit = None
        self.window_data_ratio_polyfit_sign = None

    def _check_func(self,func):
        str_func = {'count' : len, 'sum' : np.sum, 'max' : np.max, 'min' : np.min, 'median' : np.median, 'mean' : np.mean,
                    'std' : np.std}
        if isinstance(func,str):
            try:
                return str_func[func]
            except:
                print str_func
                raise KeyError('String function %s not in function list, try string above' % func)
        else:
            return func

    def _check_window_data_exist(self):
        if self.window_data is None:
            window_data = self.generate_window_data()
        else:
            window_data = self.window_data
        return window_data

    def _check_window_data_ratio_exist(self):
        if self.window_data_ratio is None:
            window_data_ratio = self.generate_window_data_ratio()
        else:
            window_data_ratio = self.window_data_ratio
        return window_data_ratio

    def generate_window_data(self):
        '''
        Generate count of aim_col during every window
        :return: pandas.DataFrame
        '''
        func = self.func
        window_data = self.data_grouped.agg(func).unstack()
        feat_name = self.feat_name
        b.add_prefix_on_col_name(window_data, feat_name)
        window_data.fillna(0, inplace=True)
        self.window_data = window_data.copy()
        return window_data

    def generate_window_data_ratio(self):
        window_data = self._check_window_data_exist()
        window_data_ratio = window_data.div(window_data.sum(axis=1), axis='index')
        feat_name = 'ratio'
        b.add_prefix_on_col_name(window_data_ratio, feat_name)
        self.window_data_ratio = window_data_ratio.copy()
        return window_data_ratio

    def generate_window_data_ratio_polyfit(self,poly_n):
        self._check_window_data_ratio_exist()
        window_data_ratio = self.window_data_ratio
        window_num = self.window_num
        reg_x = range(1, window_num+1)
        polynomial = window_data_ratio.apply(lambda y: np.polyfit(reg_x, y.values, poly_n), axis=1)
        window_data_ratio_polyfit = pd.DataFrame(index=polynomial.index)
        window_data_ratio_polyfit_sign = pd.DataFrame(index=polynomial.index)
        for i in range(poly_n+1):
            window_data_ratio_polyfit[str(i)] = polynomial.apply(lambda x: x[poly_n-i-1])
            window_data_ratio_polyfit_sign[str(i)] = polynomial.apply(lambda x: 1 if (x[poly_n-i-1]) > 0 else 0)
        b.add_prefix_on_col_name(window_data_ratio_polyfit, self.feat_name+'_ratio_poly')
        b.add_prefix_on_col_name(window_data_ratio_polyfit_sign, self.feat_name+'_ratio_poly_sign')
        self.window_data_ratio_polyfit = window_data_ratio_polyfit.copy()
        self.window_data_ratio_polyfit_sign = window_data_ratio_polyfit_sign.copy()
        return window_data_ratio_polyfit, window_data_ratio_polyfit_sign

    def generate_whole_stats(self):
        window_data = self._check_window_data_exist()
        window_data_t = window_data.T
        window_data_stats_t = window_data_t.describe()
        window_data_stats = window_data_stats_t.T
        window_data_stats.drop(columns=['count'], inplace=True)
        window_data_stats['sum'] = window_data.sum(axis=1)
        window_data_stats['skew'] = window_data.skew(axis=1)
        window_data_stats['kurt'] = window_data.kurt(axis=1)
        feat_name = self.feat_name
        b.add_prefix_on_col_name(window_data_stats, feat_name)
        self.window_data_stats = window_data_stats.copy()
        return window_data_stats


class FeatureTradeTarget(Feature):
    def __init__(self, data, cust_id_col, aim_col, stock_indu_col, func):
        col_list = [cust_id_col, aim_col, stock_indu_col]
        for col_name in [cust_id_col, aim_col, stock_indu_col]:
            b.check_col_exist(data, col_name)
        self.cust_id_col = cust_id_col
        self.aim_col = aim_col
        self.stock_indu_col = stock_indu_col
        self.data = data[col_list]
        self.data_group = data.groupbu([cust_id_col,stock_indu_col])
        self.func = _check_func(func)


    def generate_data_(self):
        data_group = self.data_group
        cust_id_col = self.cust_id_col
        aim_col = self.aim_col
        stock_indu_col = self.stock_indu_col





















