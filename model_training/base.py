#-*- coding:utf-8 -*-
# @Time : 2019/6/3
# @Author : Botao Fan

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,roc_auc_score
from sklearn import metrics
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV


def _hyper_paras_optimal_cv(model, x, y, cv_paras, other_paras, scoring='roc_auc', cv=5, verbose=3, n_jobs=3):
    '''
    Select hyperparameters using cross validation
    :param model: Model witout parameters
    :param x: Train x
    :param y: Trin y
    :param cv_paras: The parameter to select
    :param other_paras: All other given parameters
    :param scoring: The method of evaluation
    :param cv: The number of n-fold
    :param verbose:
    :param n_jobs:
    :return:
    '''
    t_start = datetime.now()
    model = model(**other_paras)
    gscv = GridSearchCV(estimator=model, param_grid=cv_paras, scoring=scoring, cv=cv, verbose=verbose, n_jobs=n_jobs)
    gscv.fit(x, y)
    t_end = datetime.now()
    time_consume = t_end - t_start
    print('参数的最佳取值：{0}'.format(gscv.best_params_))
    print('最佳模型得分:{0}'.format(gscv.best_score_))
    #print('每轮迭代运行结果:{0}'.format(gscv.grid_scores))
    print '时间:', time_consume
    return gscv.best_params_, gscv.best_score_, time_consume


class XGBoostClassifierTrain(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def cv_paras(self, other_paras, cv_paras, scoring='roc_auc', cv=5, verbose=3, n_jobs=3):
        model = XGBClassifier
        return _hyper_paras_optimal_cv(model, self.x, self.y, cv_paras, other_paras, scoring, cv, verbose, n_jobs)


def model_result_show(model, test_set):
    #Predict
    test_set_without_y = test_set.drop(columns=['y'])
    test_y = test_set[['y']].values
    test_x = test_set_without_y.values
    proba = model.predict_proba(test_x)[:, 1]
    #get roc_auc_socre
    auc_score = metrics.roc_auc_score(test_y, proba)
    y_proba_dict = {'y': test_y.reshape(1, -1)[0], 'proba': proba}
    cust_proba = pd.DataFrame(y_proba_dict, index=test_set.index)
    cust_proba.sort_values('proba', inplace=True, ascending=False)
    # feature importance
    imp_dict = {'feats': test_set_without_y.columns, 'imp': model.feature_importances_}
    feat_imp = pd.DataFrame(imp_dict)
    feat_imp.sort_values('imp', inplace=True, ascending=False)
    feat_imp.index = range(feat_imp.shape[0])

    return auc_score, cust_proba, feat_imp
