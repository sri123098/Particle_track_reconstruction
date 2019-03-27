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
from skimage.measure import compare_ssim as ssim

from model import *
from keras.backend.tensorflow_backend import set_session

def mse(A, B):
    err = np.sum((A.astype("float") - B.astype("float")) ** 2)
    err /= float(A.shape[1] * A.shape[2])
    return err

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

test_file_path = "Data/smalldataset2/d0_sparse_dataset0999.dat"
unet_weights_path = "Unet_epoch_50_batch_4.hdf5"

width = 256
ratio = 1024 / width

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.gpu_options.per_process_gpu_memory_fraction = 0.9
sess = tf.Session(config=config)
set_session(sess)

unet_model = Unet(input_size=(width,width,6), batch_normal=True)
unet_model.load_weights(unet_weights_path)

with open(test_file_path, "rb") as f:
    data = pickle.load(f)
vdata = data[0:1]

test_data = np.heaviside(np.array([map(lambda x: block_reduce(x.toarray(), block_size=(ratio,ratio), func=np.max), d.hL) for d in vdata]), 0)
gt_data = np.heaviside(np.array([map(lambda x: block_reduce(x.toarray(), block_size=(ratio,ratio), func=np.max), d.gthL) for d in vdata]), 0)
label_data = np.array([d.label for d in vdata])

# swap the axis since the data shape for CNN is (width, width, channel)
# but the original shape (num_of_images, channel, width, width)
test_data = np.swapaxes(test_data, 1, 3)

reconstruct = unet_model.predict(test_data[0:1], verbose=1)
reconstruct = np.swapaxes(reconstruct, 1, 3)
print(reconstruct.shape)
print(gt_data.shape)

###############################
#
# Save reconstructed file
#
###############################
with open("save_output.dat","wb") as f:
    pickle.dump((reconstruct, gt_data), f)

filtered = np.copy(reconstruct[0])
#PlotImage(np.swapaxes(test_data[0], 0, 2), gt_data[0], show=False)
min_threshold = 0
min_err = 99999

for threshold in np.arange(0.5, 1.01, 0.05):
    filtered[filtered < threshold] = 0
    err = mse(filtered, gt_data[0])
    if err < min_err:
        min_err = err
        min_threshold = threshold
    print("threshold:%f mse:%f" %(threshold, err))

print("min threshold:%f mse:%f" %(min_threshold, min_err))

filtered = reconstruct[0]
filtered[filtered < min_threshold] = 0
#plotLayersSinglePlot(filtered, show=False)
#plt.title("output")
#plt.legend()
#plt.show()

