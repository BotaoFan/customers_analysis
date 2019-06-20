#-*- coding:utf-8 -*-
# @Time : 2019/5/27
# @Author : Botao Fan
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from sklearn.externals import joblib

from customers_analysis.data_preprocess import preprocess as pp
from customers_analysis.data_preprocess import base as dp_base
from customers_analysis import base as b
from customers_analysis.model_training import base  as mt_base
from customers_analysis.feature_generate import feature_generate as fg
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')



if __name__ == '__main__':
    b.show_all_dataframe()
    abs_path = '/Users/fan/PycharmProjects/machine_learning/customer_loss/'
    raw_data_path = abs_path + 'raw_data/raw_data_customers_loss_1701_1704/'

    trade_date, cust_info, cust_trade = pp.script_import_and_clean(raw_data_path, datetime(2017, 1, 1), datetime(2017, 3, 31), 5)
    cust_feat = fg.script_cust_attribute_features(raw_data_path, cust_info, trade_date)
    cust_feat_stock_trade_long = fg.script_stock_trade_features(raw_data_path, cust_info, cust_trade, trade_date, long=True)
    cust_feat_stock_trade_short = fg.script_stock_trade_features(raw_data_path, cust_info, cust_trade, trade_date, long=False)
    cust_feat = pd.merge(cust_feat, cust_feat_stock_trade_long, how='left', left_index=True, right_index=True, suffixes=('', '_long'))
    cust_feat = pd.merge(cust_feat, cust_feat_stock_trade_short, how='left', left_index=True, right_index=True, suffixes=('', '_short'))

    train_count = int(cust_feat.shape[0]*0.8)
    train_set = cust_feat[:train_count]
    test_set = cust_feat[train_count:]
    train_y = train_set[['y']].values
    train_x = train_set.drop(columns=['y']).values

    #======Tune parameters======
    xgb_trainer = mt_base.XGBoostClassifierTrain(train_x, train_y)
    #Tune n_estimators
    other_paras = {'learning_rate': 0.1, 'max_depth': 5, 'min_child_weight': 1,
              'seed': 0,'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0,'random_state':42}
    cv_paras={'n_estimators': [25,50,100,200] }
    xgb_trainer.cv_paras(other_paras, cv_paras)

    #Tune min_child_weight and max_depth
    other_paras = {'learning_rate': 0.1, 'n_estimators':50,
              'seed': 0,'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0,'random_state':42}
    cv_paras={'max_depth':[3,4,5,6],'min_child_weight':[0.5,1,2,3]}
    xgb_trainer.cv_paras(other_paras, cv_paras)

    #Tune gamma
    other_paras = {'learning_rate': 0.1, 'n_estimators':100,'min_child_weight':1,'max_depth':4,
              'seed': 0,'subsample': 0.8, 'colsample_bytree': 0.8,'random_state':42}
    cv_paras={'gamma':[0,0.2,0.4,0.6,0.8]}
    xgb_trainer.cv_paras(other_paras, cv_paras)

    #Tune subsample and colsample_bytree
    other_paras = {'learning_rate': 0.1, 'n_estimators':100,'min_child_weight':1,'max_depth':4,
              'seed': 0,'gamma':4,'random_state':42}
    cv_paras={'subsample':[0.2,0.4,0.6,0.8,1],'colsample_bytree':[0.2,0.4,0.6,0.8,1]}
    xgb_trainer.cv_paras(other_paras, cv_paras)

    #Tune learning rate
    other_paras = {'n_estimators':100,'min_child_weight':1,'max_depth':4,
              'seed': 0,'gamma':4,'subsample':0.8,'colsample_bytree':0.8,'random_state':42}
    cv_paras={'learning_rate':[0.1,0.3,0.5,0.7,0.9]}
    xgb_trainer.cv_paras(other_paras, cv_paras)

    #Show test result
    paras = {'learning_rate': 0.1,'n_estimators': 50, 'min_child_weight': 1, 'max_depth': 5,
              'seed': 0, 'gamma': 4, 'subsample': 0.8, 'colsample_bytree': 0.8}

    model = XGBClassifier(**paras)
    model.fit(train_x, train_y)
    auc_score, cust_proba, feat_imp = mt_base.model_result_show(model, test_set)
    cust_proba.iloc[:1000]['y'].sum()
    joblib.dump(model, abs_path + 'model_and_data/xgboost_1701-1704.pkl')
    model = joblib.load(abs_path + 'model_and_data/xgboost_1701-1704.pkl')


#-------------
import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from sklearn.externals import joblib
from customers_analysis.data_preprocess import preprocess as pp
from customers_analysis.data_preprocess import base as dp_base
from customers_analysis import base as b
from customers_analysis.model_training import base  as mt_base
from customers_analysis.feature_generate import feature_generate as fg
warnings.filterwarnings('ignore')
b.show_all_dataframe()
abs_path = '/Users/fan/PycharmProjects/machine_learning/customer_loss/'
model = joblib.load(abs_path + 'model_and_data/xgboost_1701-1704.pkl')
raw_data_path = abs_path + 'raw_data/raw_data_customers_loss_1805_1808/'
trade_date, cust_info, cust_trade = pp.script_import_and_clean(raw_data_path, datetime(2018, 5, 2),
                                                               datetime(2018, 7, 18), 5)
cust_feat = fg.script_get_features(raw_data_path, cust_info, cust_trade, trade_date)
cust_feat.shape

test_set = cust_feat
test_y = test_set[['y']].values
test_x = test_set.drop(columns=['y']).values
auc_score, cust_proba, feat_imp = mt_base.model_result_show(model, test_set)
auc_score
cust_proba.iloc[:1000]['y'].sum()
cust_proba.iloc[:2000]['y'].sum()


