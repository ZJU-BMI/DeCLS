import numpy as np
from tensorflow.keras.models import Model
import pandas as pd
import os
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, roc_curve
import tensorflow as tf
from bayes_opt import BayesianOptimization
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import scipy.stats as stats
import os
import sys
import pickle
from data import DataSet, DataSet2, DataSetWithMask2
from sklearn.model_selection import StratifiedShuffleSplit
from lifelines.utils import concordance_index
RESULT_SAVE_DIR = 'result/'
DATA_SAVE_DIR = 'save/'
DATA_READ_DIR = 'data/'
_EPSILON = 1e-08

##### USER-DEFINED FUNCTIONS
def log(x):
    return tf.math.log(x + _EPSILON)


def div(x, y):
    return tf.math.divide(x, (y + _EPSILON))




def import_mimic_data():
    with open(DATA_READ_DIR + 'mimic_time_data.pkl', 'rb') as f:
        data = pickle.load(f)
    print(data.columns)
    # c = data.columns
    # c=pd.DataFrame(c)
    # c.to_excel('d:/c.xlsx')
    return data


def import_eicu_data():
    with open(DATA_READ_DIR + 'eicu_dataset.pkl', 'rb') as f:
        data = pickle.load(f)
    return data


def partial_log_likelihood(prediction, t, y):
    """
    calculate cox loss
    :param prediction: prediction of model
    :param t: event happen at the 't'th day
    :param y: true label
    :return:
    """
    risk = tf.reshape(prediction, [-1])
    time = tf.reshape(t, [-1])
    E = tf.reshape(y, [-1])
    sort_idx = tf.argsort(time, direction='DESCENDING')
    E = tf.gather(E, sort_idx)
    risk = tf.gather(risk, sort_idx)
    hazard_ratio = tf.exp(risk)
    log_risk = tf.math.log(tf.cumsum(hazard_ratio))
    uncensored_likelihood = risk - log_risk
    censored_likelihood = tf.multiply(uncensored_likelihood, E)
    neg_likelihood = -tf.reduce_sum(censored_likelihood) * 0.01
    return neg_likelihood


def calculate_score(y_label, y_prediction, print_flag=False):
    """
    :param y_label: true label
    :param y_prediction: prediction of model
    :param print_flag: print pr not
    :return: auc, precision, recall, f_score, accuracy
    """
    try:
        auc = roc_auc_score(y_label, y_prediction)
        fpr, tpr, thresholds = roc_curve(y_label, y_prediction)
        threshold = thresholds[np.argmax(tpr - fpr)]
        y_pred_label = (y_prediction >= threshold)
        precision = precision_score(y_label, y_pred_label)
        recall = recall_score(y_label, y_pred_label)
        f_score = f1_score(y_label, y_pred_label)
        accuracy = accuracy_score(y_label, y_pred_label)
        if print_flag:
            print('auc:{} precision:{} recall:{} f_score:{} accuracy:{}'.format(auc, precision, recall, f_score,
                                                                                accuracy))
    except:
        return 0,0,0,0,0
    return y_pred_label, auc, precision, recall, f_score, accuracy


