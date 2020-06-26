#!/usr/bin/env python
# coding: utf-8

# ## Алгоритм оптимизации суперпозиции моделей

# In[234]:


import io
import requests
import seaborn as sns
import pickle
import os
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score


# In[232]:


import numpy as np
np.random.seed(seed=42)
import pandas as pd
import itertools


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import copy
import math

import matplotlib
import matplotlib.pyplot as plt


# In[3]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold

pd.set_option('display.float_format','{:,.5f}'.format)
pd.set_option('display.max_columns', None)
from IPython import display


# In[4]:


# отключим предупреждения Anaconda
import warnings
warnings.simplefilter('ignore')

# будем отображать графики прямо в jupyter'e
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt
#графики в svg выглядят более четкими
# %config InlineBackend.figure_format = 'svg' 
plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 12
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24
plt.rcParams['legend.fontsize'] = 24
plt.rcParams['axes.titlesize'] = 36
plt.rcParams['axes.labelsize'] = 24
plt.rcParams['figure.figsize'] = [7, 5]
### Сохранение изображения ###
plt.savefig('1.svg') # Поддерживаемые форматы: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff


# In[5]:


from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal as mvn


# In[6]:


from utils import *


# ## Логистическая регрессия. Алгоритм Ньютона-Рафсона. В оптимизации участвует ковариационная матрица признаков.

# In[7]:


def preprocess(arr):
    # Adds fictitious feature equal to 1
    
    ones = np.array([[1]*arr.shape[0]]).T
    arr_b = np.concatenate((ones, arr), axis=1)
    return arr_b


# In[8]:


def sigmoid(z):
    # Activation function used to map any real value between 0 and 1
    
    sig = 1 / (1 + np.exp(-z))     # Define sigmoid function
    sig = np.minimum(sig, 0.99999999999999999999999999)  # Set upper bound
    sig = np.maximum(sig, 0.00000000000000000000000001)  # Set lower bound
    return sig 


# In[9]:


def cross_entropy(T, Y):
    # Computes the cost function for all the training samples
    
    E = 0
    for i in range(T.shape[0]):
        if T[i] == 1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1 - Y[i])
    return E/T.shape[0]


# In[88]:


import random


# In[184]:


def step_of_NR_algo(Y, w, Xb, T_):
    # Computes one step of Newton-Raphson method
    
#     SGD
#     k = 128
#     indx = random.sample(range(len(Xb)), k)
#     Y = Y[indx]
#     Xb = Xb[indx]
#     T_ = T_[indx]
    
    b = Y*(1-Y)
    B = np.diag(b)

    inverse_of_A = np.dot(w.reshape(Xb.shape[1],1), w.reshape(1,Xb.shape[1]))/Xb.shape[1]
    cov_w_matrix = np.linalg.pinv(inverse_of_A)
    var_w = cov_w_matrix.diagonal()

    Xt = np.transpose(Xb)
    XtB = Xt.dot(B)
    XtBX = XtB.dot(Xb)
#     print(i)
#     print(w)
#     print(Y)
    #inverse_of_H = np.linalg.pinv(XtBX + var_w*1e-19)
    inverse_of_H = np.linalg.pinv(XtBX + var_w)
#     pdb.set_trace()

#     Y = Y.reshape(-1,1)
#     T_ = T_.reshape(-1,1)
    derivative_of_logloss = np.dot(Xb.T, (Y-T_))

    w = w - inverse_of_H.dot(derivative_of_logloss + var_w*w)
    
    return w, var_w, b


# In[185]:


def filter_columns(Xb_test, cols):
    # Returns array filtered by given columns
    
    Xb_test = pd.DataFrame(data=Xb_test, columns = [str(i) for i in range(Xb_test.shape[1])])
    filtered_Xb_test = np.array(Xb_test[cols])
    return filtered_Xb_test


# In[186]:


def predict_estimates(X_test, w, feat_obj_selection=False, cols=None):
    # Returns probabilities
    
    Xb_test = preprocess(X_test)
    if feat_obj_selection:
        Xb_test = filter_columns(Xb_test=Xb_test, cols=cols)
    z_test = Xb_test.dot(w)
    probability_estimates = sigmoid(z_test)
    return probability_estimates


# In[187]:


class LogisticRegressionModel(object):
    """A logistic regression model for fitting and predicting binary response data.
    
    Attributes:
        X: the predictor matrix,
        T: the response vector 
    """    

    def __init__(self):
        self.w = None
        
    def set_weights(self, w):
        self.w = w
    
    def fit(self, X, T, iterations=12):

        X_ = copy.deepcopy(X)
        T_ = copy.deepcopy(T)

        N, D = X_.shape

        Xb = preprocess(X_)
        
        if self.w is None:
            self.w = np.ones(D+1)*math.log(T_.mean()/(1-T_.mean()))
#         w = np.ones(D+1)*math.log(T_.mean()/(1-T_.mean()))
        
        list_w = []
        list_var_w = []
        list_var_obj = []
        costs = []
        self.converged = False

        for i in range(iterations):

            Y_hat = sigmoid(Xb.dot(self.w))
            Y = Y_hat
            Y[Y==1.0] = 0.999999
            Y[Y==0.0] = 0.000001

            costs += [cross_entropy(T_, Y)]
            
            if i>1:
                if not self.converged and abs(costs[-1]-costs[-2])<.000001:
                    self.converged = True
                    self.converged_k = i+1

            self.w, var_w, b = step_of_NR_algo(Y, self.w, Xb, T_)

           # w = w - learning_rate *(np.dot(Xb.T, (Y-T)) + var_w*w*1e-19)

           # inverse_of_A = np.dot(w.reshape(D+1,1), w.reshape(1,D+1))/D
           # var_w = np.linalg.pinv(inverse_of_A).diagonal()
            list_w.append(self.w)
            list_var_w.append(var_w)
            list_var_obj.append(Y*(1-Y))

        self.list_w = list_w
        self.list_var_w = list_var_w 
        self.list_var_obj = list_var_obj
        self.costs = costs
        
        if not self.converged:
            print('Недостаточно итераций для сходимости алгоритма.')
        else:
            print('\nАлгоритм сошелся за {} итераций'.format(self.converged_k))

        return self
 
    def predict(self, X):
        return predict_estimates(X, self.w, feat_obj_selection=False, cols=None)
    
    def cross_entropy(self, y, p):
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    
    #Вспомогательные методы
    def plot_weights(self):
        plt.scatter(np.arange(len(self.w)), self.w, label='w map')
        plt.legend()
    
    def plot_cross_entropy(self):
        plt.plot(np.arange(len(self.costs)),self.costs)
        plt.xlabel('Iteration number')
        plt.ylabel('Cross-Entropy')
        plt.title('Зависимость S от номера итерации')   
        
    def make_report(self):
        """
        Prints a formatted table of the model coefficients 
        """
        
        if not self.converged:
            print('\nНедостаточно итераций для сходимости алгоритма.')
        else:
            print('\nАлгоритм сошелся за {} итераций'.format(self.converged_k))


# ## Реализация Н-Р с отбором признаков по дисперсиям и метрическим блоком

# In[188]:


def select_objects(Y, obj_threshold, Xb, T):
    # Returns objects with low variance
    
    b_temp = Y*(1-Y)
    mask = b_temp<obj_threshold
    Xb = Xb[mask]
    Y = Y[mask]
    T = T[mask]
    
    return Xb, Y, T


# In[189]:


def select_features(w, Xb, feat_threshold, cols):
    # Returns features with low variance
    
    inverse_of_A = np.dot(w.reshape(Xb.shape[1],1), w.reshape(1,Xb.shape[1]))/Xb.shape[1]
#     cov_w_matrix = np.linalg.pinv(inverse_of_A)
    try:
        cov_w_matrix = np.linalg.inv(inverse_of_A)
    except np.linalg.LinAlgError as err:
        if 'Singular matrix' in str(err):
            cov_w_matrix = np.linalg.pinv(inverse_of_A)
    var_w = cov_w_matrix.diagonal()

    print('Дисперсии весов')
    print(var_w)
    print('\nСписок выбранных на текущей итерации признаков')
    print(cols)
    print('\nРазмер матрицы Плана на текущей итерации')
    print(Xb.shape)
    mask2 = np.abs(var_w)<feat_threshold
    cols = [col for col, mask in zip(cols, mask2) if mask]
    w = w[mask2]
    Xb = Xb[:, mask2]
    
    return cols, w, Xb


# In[190]:


def my_kmeans(n_clusters, X):
    
    # В scipy есть замечательная функция, которая считает расстояния
    # между парами точек из двух массивов, подающихся ей на вход
    from scipy.spatial.distance import cdist

    # Прибьём рандомность и насыпем случайные центроиды для начала
    np.random.seed(seed=42)
    centroids = X[10:10+n_clusters]
#     centroids = np.random.normal(loc=0.0, scale=1., size=5*X.shape[1])
#     centroids = centroids.reshape((5, X.shape[1]))

    cent_history = []
    cent_history.append(centroids)

    for i in range(10):
        # Считаем расстояния от наблюдений до центроид
        distances = cdist(X, centroids)
        # Смотрим, до какой центроиде каждой точке ближе всего
        labels = distances.argmin(axis=1)

        centroids = centroids.copy()
        for i in range(n_clusters):
            # Положим в каждую новую центроиду геометрический центр её точек
            centroids[i, :] = np.mean(X[labels == i, :], axis=0)

        cent_history.append(centroids)
        
    return cent_history[-1]


# In[191]:


def predict_estimates_with_MF(X_test, w, cols, cluster_centers, metric_mask):
    if len(cluster_centers) == 0:
        probab_estimates = predict_estimates(X_test, w, feat_obj_selection=True, cols=cols)
    else:
        Xb_test = preprocess(X_test)
        not_metric_features = [x for x in cols if 'cluster_' not in x]
        Xb_test = filter_columns(Xb_test=Xb_test, cols=not_metric_features)
        metric_features_for_test = pairwise_distances(Xb_test, cluster_centers)[:, metric_mask]
        Xb_test_new = np.hstack((Xb_test, metric_features_for_test))
        z_test = Xb_test_new.dot(w)
        probab_estimates = sigmoid(z_test)
    return probab_estimates


# In[192]:


def metric_block(Xb, cols, n_clusters, i, algo_clust='km_sklearn'):
    not_metric_features = ['cluster_' not in x for x in cols]
    
    if algo_clust=='km_simple':
        new_centroids = my_kmeans(n_clusters, Xb[:, not_metric_features])
    
    elif algo_clust=='km_sklearn':
        kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(Xb[:, not_metric_features])

        bc = np.bincount(kmeans.labels_)
        new_order = sorted(range(n_clusters), key=lambda x: bc[x])
        new_centroids = np.array([kmeans.cluster_centers_[i] for i in new_order])
    
    elif algo_clust=='gmm':
        not_metric_features = ['cluster_' not in x for x in cols]
        data = Xb[:, not_metric_features]
        gmm = GaussianMixture(n_components=n_clusters, covariance_type='full').fit(data)
        prediction_gmm = gmm.predict(data)
        probs = gmm.predict_proba(data)

        centers = np.zeros((n_clusters, data.shape[1]))
        for i in range(n_clusters):
            density = mvn(cov=gmm.covariances_[i], mean=gmm.means_[i]).logpdf(data)
            centers[i, :] = data[np.argmax(density)]
        new_centroids=centers
    
    metric_features = pairwise_distances(Xb[:, not_metric_features], new_centroids)
    scaler = StandardScaler()
    metric_features = scaler.fit_transform(metric_features)

    print('\nМассив метрических признаков на {}-й итерации'.format(i))
    print(metric_features)

    cluster_centers = new_centroids

    new_metric_cols = ['cluster_'+str(x) for x in range(metric_features.shape[1])]

    # Кластера из предыдущей итерации
    metric_cols = [x for x in cols if 'cluster_' in x]

    # metric_cols = new_metric_cols[metric_mask]
    metric_mask = [x in metric_cols for x in new_metric_cols]

    # Маска длины cols для кластеров, True только там, где кластера.
    # Позволяет выделить измененные на предыдущей итерации метрические признаки
    metric_all_mask = ['cluster_' in x for x in cols]
    Xb[:, metric_all_mask] = metric_features[:, metric_mask]
    
    return metric_mask, cluster_centers


# In[193]:


class SuperpositionModel(LogisticRegressionModel):
    """A logistic regression model for fitting and predicting binary response data.
    
    Attributes:
        X: the predictor matrix,
        T: the response vector 
    """    
    
    def __init__(self):
        # Необходимо вызвать метод инициализации родителя.
        # В Python 3.x это делается при помощи функции super()
        super().__init__()
        
    def fit(self, X, T, feat_threshold, n_clusters, iterations=10, obj_threshold=100, algo_clust='km_sklearn'):

        X_ = copy.deepcopy(X)
        T_ = copy.deepcopy(T)

        N, D = X_.shape

        Xb = preprocess(X_)
        Xb = np.hstack((Xb, np.zeros((Xb.shape[0], n_clusters))))

        cols = [str(i) for i in range(D + 1)] + ["cluster_" + str(x) for x in range(n_clusters)]

        if self.w is None:
#             self.w = np.random.randn(D + 1 + n_clusters)*0.02
#             self.w = np.ones(D+1 + n_clusters)*math.log(T.mean()/(1-T.mean()))*0.02 + np.random.randn(D + 1 + n_clusters)*0.02
#             self.w = np.ones(D+1 + n_clusters)*-0.07710948
            self.w = np.random.uniform(-0.07, 0.008, D + 1 + n_clusters)
        print(self.w)
        list_w = []
        list_var_w = []
        list_var_obj = []
        costs = []
        cluster_centers = []
        metric_mask = []
        self.converged = False

        for i in range(iterations):
            print('\n{}-я итерация'.format(i))
            print('\nВеса')
            print(self.w)

            if n_clusters == 0:
                cluster_centers = []
                metric_mask = []

            if i!=0:
                cols, self.w, Xb = select_features(self.w, Xb, feat_threshold, cols)
                if n_clusters != 0:
                    metric_mask, cluster_centers = metric_block(Xb, cols, n_clusters, i, algo_clust)

            # Модель
            Y = sigmoid(Xb.dot(self.w))

            print('\nСкоры объектов на {}-й итерации'.format(i))
            print(Y)

            Xb, Y, T_ = select_objects(Y, obj_threshold, Xb, T_)

            costs += [cross_entropy(T_, Y)]
            
            if i>1:
                if not self.converged and abs(costs[-1]-costs[-2])<.000001:
                    self.converged = True
                    self.converged_k = i+1

            self.w, var_w, b = step_of_NR_algo(Y, self.w, Xb, T_)
            
            list_w.append(self.w)
            list_var_w.append(var_w)
            list_var_obj.append(Y*(1-Y))

            self.list_w = list_w
            self.list_var_w = list_var_w 
            self.list_var_obj = list_var_obj
            self.costs = costs
            self.cols = cols
            self.metric_mask = metric_mask
            self.cluster_centers = cluster_centers

            if not self.converged:
                print('Недостаточно итераций для сходимости алгоритма.')
            else:
                print('\nАлгоритм сошелся за {} итераций'.format(self.converged_k))

        return self
    
    def get_costs(self):
        return self.costs
    
    def get_cols(self):
        return self.cols
    
    def get_weights(self):
        return self.w
    
    def predict(self, X):
        return predict_estimates_with_MF(X, self.w, self.cols, self.cluster_centers, self.metric_mask)  


# In[194]:


def plot_feature_importance(w, cols):
    feature_importance = abs(w)
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5

    featfig = plt.figure(figsize=(15,13))
    featax = featfig.add_subplot(1, 1, 1)
    featax.barh(pos, feature_importance[sorted_idx], align='center', color='#27AE60')
    featax.set_yticks(pos)
    featax.set_yticklabels(np.array(cols)[sorted_idx])
    featax.set_xlabel('Relative Feature Importance')

    plt.tight_layout()
