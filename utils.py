# identify outliers with standard deviation

import pandas as pd
import numpy as np
from numpy.random import seed
from numpy.random import randn
from numpy import mean
from numpy import std
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def remove_outliers(data):
    """
    A function for removing outliers from dataset.
    :param data: dataset
    """
    
    # calculate summary statistics

    data_mean = []
    data_std = []
    for i in range(len(data.columns)):
        data_mean.append(mean(data[data.columns[i]]))
        data_std.append(std(data[data.columns[i]]))
    
    stats = pd.DataFrame({'data_mean': data_mean,
                      'data_std': data_std,
                      'two_sigma': np.array(data_std)*2,
                      'three_sigma': np.array(data_std)*3}, index=data.columns)
    
    # identify outliers

    cut_off = np.array(stats.two_sigma)
    stats['lower'], stats['upper'] = np.array(data_mean) - cut_off, np.array(data_mean) + cut_off

    # identify outliers

    outliers_set = set()
    for feat in stats.index:
        low = stats.loc[feat, 'lower']
        up = stats.loc[feat, 'upper']
        outliers_set = outliers_set | set(data[(data[feat] < low) | (data[feat] > up)].index)

    outliers = data[data.index.isin(outliers_set)]
    number_outs = outliers.shape[0]
    
    # remove outliers

    outliers_removed = data[~data.index.isin(outliers_set)]
    number_outs_rem = outliers_removed.shape[0]

    
    return list(outliers_removed.index), list(outliers.index)



def top_15_features(df):
    """
    A function for choosing top_15 the most important features for logistic regression.
    this function needs feature scaling before
    :param df: dataset
    """
    
    
    df = df.copy()
    y = df['TARGET']
    df.drop('TARGET', axis=1, inplace=True)
    
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25, random_state=42)
    
    clf = LogisticRegression(random_state=0).fit(X_train, y_train)
    y_scores = clf.predict_proba(X_test)[:, 1]
    
    feature_importance = abs(clf.coef_[0])
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    feat_importances = pd.DataFrame({'features': X_test.columns[sorted_idx], 'weights': feature_importance[sorted_idx]})
    feat_importances.sort_values('weights', ascending=False, inplace=True)
    A_cal_0 = list(feat_importances.features.iloc[0:15,])

    return roc_auc_score(y_test, y_scores), A_cal_0


def mean_value_imputer(data):
    """
    A function for filling missing values in dataset with mean value for each feature.
    :param data: dataset
    """
    
    
    X = np.array(data)
    mask = X != X

    for col in range(X.shape[1]):
        X[mask[:, col], col] = np.mean(X[~mask[:, col], col])
    
    return X


def plot_roc_cur(fper, tper, title):  
    plt.plot(fper, tper, color='#27AE60', label='ROC')
    plt.plot([0, 1], [0, 1], color='#AE2775', linestyle='--', label='random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()