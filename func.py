# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 12:04:28 2020

@author: Aprotckii_MV
"""

import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


import scipy

def proportions_diff_z_stat_ind(sample1, sample2):
    n1 = len(sample1)
    n2 = len(sample2)
    
    p1 = float(sum(sample1)) / n1
    p2 = float(sum(sample2)) / n2 
    P = float(p1*n1 + p2*n2) / (n1 + n2)
    
    return (p1 - p2) / np.sqrt(P * (1 - P) * (1. / n1 + 1. / n2))

def proportions_diff_z_test(z_stat, alternative = 'two-sided'):
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError("alternative not recognized\n"
                         "should be 'two-sided', 'less' or 'greater'")
    
    if alternative == 'two-sided':
        return 2 * (1 - scipy.stats.norm.cdf(np.abs(z_stat)))
    
    if alternative == 'less':
        return scipy.stats.norm.cdf(z_stat)

    if alternative == 'greater':
        return 1 - scipy.stats.norm.cdf(z_stat)


def classification(X, y, X_na, names, classifiers, parameter_grid):
    # Х - трейн дата, y - тест дата, X_na - Х для классификации,
    #classifiers - классификаторы, parameter_grid - RF гиперпараметры

    clf = RandomForestClassifier(random_state=1, class_weight='balanced_subsample')
    grid_searcher = GridSearchCV(clf, parameter_grid)
    grid_searcher.fit(X, y)
    clf_best = grid_searcher.best_estimator_
    y_hat=clf_best.predict(X_na)          
    max_score=grid_searcher.best_score_
    best_clf='RF_corr'
    d={best_clf:max_score}
    
    for name, clf in zip(names, classifiers):
        d[name] = cross_val_score(clf, X, y, cv=5).mean()
        if d[name] > max_score:
            max_score=d[name]
            best_clf = name
            clf.fit(X, y)
            y_hat=clf.predict(X_na)
    print('best clf: ',best_clf, ' best score: ', max_score)                
    return y_hat.reshape(-1,1)