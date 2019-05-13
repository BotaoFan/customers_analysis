#-*- coding:utf-8 -*-
# @Time : 2019/5/12
# @Author : Botao Fan
import pandas as pd

from . import base as b
from .data_preprocess import base as dp_base
from .feature_generate import feature_generate as fg


if __name__=="__main__":
    #Prepare cust_feats
    hdf=pd.HDFStore('feature_data.h5')
    cust_info=hdf['cust_info']
    cust_trade=hdf['cust_trade']
    hdf.close()
    cust_info.set_index('khh',drop=False,inplace=True)
    cust_feats=cust_info[['khh']]

    #cust_age
    feat_cust_age=fg.FeatureAge()
    cust_feats['cust_age'] = feat_cust_age.generate_from_num_birthday(cust_info['birthday'])
    del feat_cust_age
    #cust_age_bin
    feat_age_bin=fg.FeatureAgeBin()
    cust_feats['cust_age_bin'] = feat_age_bin.generate_from_age(cust_feats['cust_age'])
    del feat_age_bin
    #open_years
    feat_age = fg.FeatureAge()
    cust_feats['open_years'] = feat_age.generate_from_num_birthday(cust_info['khrq'])
    del feat_age
    #open_years_bin
    feat_age_bin=fg.FeatureAgeBin()
    cust_feats['open_years_bin'] = feat_age_bin.generate_from_age(cust_feats['open_years'],bins=range(0,33,3))
    del feat_age_bin
    #open_area
    area_dict={u'上海':2,u'北京':2,u'广东':2,u'浙江':1,u'江苏':1,u'山东':1,u'福建':1,
               u'湖南':0,u'湖北':0,u'江西':0,u'河南':0,u'广西':0,u'辽宁':0,u'黑龙江':0,
               u'山西':0,u'天津':0,u'河北':0,u'云南':0,u'安徽':0,u'吉林':0,u'内蒙':0,u'青海':0,
               u'四川':0,u'重庆':0,u'贵州':0,u'新疆':0,u'陕西':0,u'甘肃':0,u'海南':0,u'宁夏':0}
    area_se=dp_base.value_map(cust_info['area'], area_dict)
    feat_area_economy = fg.FeatureOneHot()
    area_economy_df = feat_area_economy.generate(area_se, 'area_economy')
    cust_feats = pd.merge(cust_feats, area_economy_df, how='left', left_index=True, right_index=True)
    feat_extract_col = fg.FeatureExtractColumn(cust_info)
    cust_feats['cust_start_asset']=feat_extract_col.generate('start_jyzc','cust_start_asset')
    cust_feats['cust_start_asset_ln']=feat_extract_col.generate_log1p('start_jyzc','cust_start_asset')






