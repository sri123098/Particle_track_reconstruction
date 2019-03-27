import json
from pprint import pprint
import pickle
import matplotlib.pyplot as plt
from plotLayer import *
from preprocess import *

import os
import time
import numpy as np
from scipy import sparse as sp
from skimage.measure import block_reduce

from model import *
from keras.backend.tensorflow_backend import set_session
from keras.datasets import cifar100

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

(X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='fine')
# 32 x 32 x 1

X_train = X_train.astype('float32')[:3]
#X_test = X_test.astype('float32')[:3]

X_train = np.round(X_train/255.0)

X_train = np.concatenate(([X_train] * 2), 3)
X_train = np.concatenate(([X_train] * 4), 2)
X_train = np.concatenate(([X_train] * 4), 1)

width = 128
data_shape=(width, width, 6)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.gpu_options.per_process_gpu_memory_fraction = 0.9
sess = tf.Session(config=config)
set_session(sess)

unet_model = Unet(input_size=(width,width,6), batch_normal=True)
unet_model.load_weights("mnist_epoch_20_batch_2.hdf5")

reconstruct = unet_model.predict(X_train[0:1], verbose=1)
reconstruct = np.swapaxes(reconstruct, 1, 3)
X_train = np.swapaxes(X_train, 1, 3)
print(reconstruct.shape)

filtered = reconstruct[0]
PlotImage(X_train[0], X_train[0], show=False)
filtered[filtered < 0.5] = 0
PlotImage(filtered, X_train[0], show=False)
plt.show()
