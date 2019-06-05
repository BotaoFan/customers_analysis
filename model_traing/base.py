#-*- coding:utf-8 -*-
# @Time : 2019/6/3
# @Author : Botao Fan

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,roc_auc_score
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
    model = model(**other_paras)
    gscv = GridSearchCV(estimator=model, param_grid=cv_paras, scoring=scoring, cv=cv, verbose=verbose, n_jobs=n_jobs)
    gscv.fit(x, y)
    print('参数的最佳取值：{0}'.format(gscv.best_params_))
    print('最佳模型得分:{0}'.format(gscv.best_score_))
    print('每轮迭代运行结果:{0}'.format(gscv.grid_scores_))
    return gscv.best_params_, gscv.best_score_, gscv.grid_scores


class XGBoostClassifierTrain(object):
    def __init__(self, x, y, paras):
        self.x = x
        self.y = y
        self.paras = paras

    def cv_paras(self, cv_paras, scoring = 'roc_auc', cv=5, verbose=3, n_jobs=3):
        other_paras = self.pares
        for key in cv_paras.keys():
            other_paras.pop(key)
        model = XGBClassifier()
        return _hyper_paras_optimal_cv(model, self.x, self.y, cv_paras, other_paras, scoring, cv, verbose, n_jobs)

    def change_paras(self, paras):
        self.paras

