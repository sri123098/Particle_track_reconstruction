from __future__ import division
import json
from pprint import pprint
import pickle
import matplotlib.pyplot as plt
from plotLayer import *
from preprocess import *

import os
import time
import numpy as np
import glob
from scipy import sparse as sp
from skimage.measure import block_reduce
from sklearn.metrics import roc_curve, auc, roc_auc_score, classification_report

from model import *
from keras.backend.tensorflow_backend import set_session

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def Classify_Rate(y, y_hat):
    # True Positive (TP): we predict a label of 1 (positive), and the true label is 1.
    TP = np.sum(np.logical_and(y_hat == 1, y == 1))
    # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
    TN = np.sum(np.logical_and(y_hat == 0, y == 0))
    # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
    FP = np.sum(np.logical_and(y_hat == 1, y == 0))
    # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
    FN = np.sum(np.logical_and(y_hat == 0, y == 1))
    return TP, TN, FP, FN


def Predict(model, x, y_threshold=None):
    y_hat = model.predict(x)

    if y_threshold :
        y_hat[y_hat < y_threshold] = 0
        y_hat[y_hat >= y_threshold] = 1
    return y_hat

def PlotROC(y, y_hat):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(y, y_hat)
        roc_auc[i] = auc(fpr[i], tpr[i])
    print ("roc_auc_score:%f" %roc_auc_score(y, y_hat))
    plt.figure()
    plt.plot(fpr[1], tpr[1])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.show()

def PreprocessData(datapath, width=256, channel=6):
    ratio = 1024 // width
    with open(datapath, "rb") as f:
        data = pickle.load(f)
    x = np.heaviside(np.array([map(lambda x: block_reduce(x.toarray(), block_size=(ratio,ratio), func=np.max), d.hL) for d in data]), 0)
    x = np.swapaxes(x, 1, 3)
    y = np.array([d.label for d in data])
    return x, y


if __name__ == "__main__":
    width = 256
    channel = 6
    #classify_weights_path = "Classify_epoch_50_batch_4.hdf5"
    classify_weights_path = "Classify_epoch_100_batch_4.hdf5"


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess = tf.Session(config=config)
    set_session(sess)

    classify_model = Encoder_Classify(input_size=(width,width,6), batch_normal=True)
    classify_model.load_weights(classify_weights_path)

    d0_path = "Data/1stDataset/d0*"
    false_path = "Data/1stDataset/false*"
    d0 = np.sort(glob.glob(d0_path))
    f0 = np.sort(glob.glob(false_path))

    y_total = []
    y_hat_total = []
    for i in range(100):
        x1, y1 = PreprocessData(d0[i], width=width, channel=channel)
        x2, y2 = PreprocessData(f0[i], width=width, channel=channel)
        x = np.concatenate((x1, x2), axis=0)
        y = np.concatenate((y1, y2), axis=0)

        y_hat = Predict(classify_model, x)
        y_total += [y]
        y_hat_total += [y_hat]
        print("y_hat mean:%f median:%f" %(np.mean(y_hat), np.median(y_hat)))
        y_hat[y_hat < np.median(y_hat)] = 0
        y_hat[y_hat >= np.median(y_hat)] = 1

        TP, TN, FP, FN = Classify_Rate(y, y_hat)
        print("Sensitive TP/(TP+FN) : %f" %(TP / (TP + FN)))
        print(classification_report(y, y_hat))

    y = np.concatenate(y_total)
    y_hat = np.concatenate(y_hat_total)
    print(y.shape)
    print(y_hat.shape)
    print("y_hat mean:%f median:%f" %(np.mean(y_hat), np.median(y_hat)))
    y_hat[y_hat < np.median(y_hat)] = 0
    y_hat[y_hat >= np.median(y_hat)] = 1
    TP, TN, FP, FN = Classify_Rate(y, y_hat)
    print("Sensitive TP/(TP+FN) : %f" %(TP / (TP + FN)))
    print(classification_report(y, y_hat))

