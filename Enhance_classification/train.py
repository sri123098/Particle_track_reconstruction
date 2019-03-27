from model import *
from preprocess import *
import os
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
#from tensorflow import logging
#logging.set_verbosity(logging.INFO)

import glob
import time
import numpy as np
from scipy.ndimage import zoom
from skimage.measure import block_reduce
from scipy import sparse as sp
import keras
from keras.backend.tensorflow_backend import set_session


class Data_Generator(keras.utils.Sequence):
    def __init__(self, in_dir, batch_size=1, data_shape=(512, 512, 6), shuffle=True, output='image'):
        self.path = in_dir
        self.cached_data = {}
        self.file_list = glob.glob(in_dir + "/*")
        self.file_size = len(self.file_list)
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.data_shape = data_shape
        self.ratio = 1024 / data_shape[0]
        self.output = output

    def __data_generation(self, index):
        batch_in = np.zeros((self.batch_size,) + self.data_shape)
        if self.output == 'image':
            batch_out = np.zeros((self.batch_size,) + self.data_shape)
        elif self.output == 'label':
            batch_out = np.zeros(self.batch_size)

        for i in index:
            if os.path.basename(self.file_list[i]) in self.cached_data:
                data = self.cached_data[self.file_list[i]]
            else:
                with open(self.file_list[i], "rb") as f:
                    data = pickle.load(f)
                    self.cached_data[self.file_list[i]] = data

            # randomly pick one event from random file
            ind = np.random.randint(0, len(data))
            print(ind)
            train_data = map(lambda x: block_reduce(x.toarray(), block_size=(self.ratio,self.ratio), func=np.max), data[ind].hL)
            train_data = np.swapaxes(train_data, 0, 2)
            batch_in[i] = train_data

            if self.output == 'image':
                gt_data = map(lambda x: block_reduce(x.toarray(), block_size=(self.ratio,self.ratio), func=np.max), data[ind].gthL)
                gt_data = np.swapaxes(gt_data, 0, 2)
            elif self.output == 'label':
                gt_data = data[ind].label
            batch_out[i] = gt_data

        return batch_in, batch_out

    def __len__(self):
        return (self.file_size) // self.batch_size

    def __getitem__(self, ind):
        #index = self.index[ind * self.batch_size:(ind + 1) * self.batch_size]
        index = np.random.randint(0, self.file_size, self.batch_size)
        print(ind)
        return self.__data_generation(index)

class My_Generator():
    def __init__(self, in_dir, batch_size, data_shape, output='image'):
        self.path = in_dir
        self.cached_data = {}
        self.file_list = glob.glob(in_dir + "/*")
        self.batch_size = batch_size
        self.ratio = 1024 / data_shape[0]
        self.data_shape = data_shape
        self.output = output

    def Generator(self):
        np.random.shuffle(self.file_list)
        batch_in = np.zeros((self.batch_size,) + self.data_shape)

        if self.output == 'image':
            batch_out = np.zeros((self.batch_size,) + self.data_shape)
        elif self.output == 'label':
            batch_out = np.zeros(self.batch_size)

        while True:
            for i in range(self.batch_size):
                f_ind = np.random.randint(0, len(self.file_list))
                if os.path.basename(self.file_list[f_ind]) in self.cached_data:
                    data = self.cached_data[self.file_list[f_ind]]
                else:
                    with open(self.file_list[f_ind], "rb") as f:
                        data = pickle.load(f)
                        self.cached_data[self.file_list[f_ind]] = data
                ind = np.random.randint(0, len(data))

                train_data = map(lambda x: block_reduce(x.toarray(), block_size=(self.ratio,self.ratio), func=np.max), data[ind].hL)
                train_data = np.swapaxes(train_data, 0, 2)
                batch_in[i] = np.heaviside(train_data, 0)

                if self.output == 'image':
                    gt_data = map(lambda x: block_reduce(x.toarray(), block_size=(self.ratio,self.ratio), func=np.max), data[ind].gthL)
                    gt_data = np.swapaxes(gt_data, 0, 2)
                elif self.output == 'label':
                    gt_data = np.array(data[ind].label)
                batch_out[i] = np.heaviside(gt_data, 0)

            yield batch_in, batch_out


d0_path = "Data/1stDataset/d0*"
false_path = "Data/1stDataset/false*"

width = 256
data_shape=(width, width, 6)
batch_size = 5
epochs = 100
ckname = "epoch_" + str(epochs) + "_batch_" + str(batch_size) + ".hdf5"


##############################
#
# Initialize session
#
##############################

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.gpu_options.per_process_gpu_memory_fraction = 0.9
sess = tf.Session(config=config)
set_session(sess)

##############################
#
# Train autoencoder
#
##############################

model_Unet = Unet(input_size=data_shape, batch_normal=True, loss='binary_crossentropy')
unet_model_checkpoint = ModelCheckpoint("Unet_" + ckname, monitor='loss', verbose=0, save_best_only=True)
unet_callbacks = [unet_model_checkpoint]

model_Unet.load_weights("../Unet_epoch_15_batch_5.hdf5")

#print("Train autoencoder")
#mygen_image = My_Generator("Data/smallDataset", batch_size=batch_size, data_shape=data_shape, output='image')
#model_Unet.fit_generator(mygen_image.Generator(), epochs=epochs, steps_per_epoch=50, verbose=1, callbacks=unet_callbacks)



######################################################################
#
# Train classifier with first 24 layers fixed and identical to encoder
#
######################################################################

print("Train classifier")
model_Classify = Encoder_Classify(input_size=data_shape, batch_normal=True, loss='binary_crossentropy')
classify_model_checkpoint = ModelCheckpoint("Classify_" + ckname, monitor='loss', verbose=0, save_best_only=True)
classify_callbacks = [classify_model_checkpoint]

mygen_label = My_Generator("Data/smallDataset", batch_size=batch_size, data_shape=data_shape, output='label')

print("Load weights 0:24 from Unet %d to Classify %d" %(len(model_Unet.layers), len(model_Classify.layers)))
for l1,l2 in zip(model_Classify.layers[:23],model_Unet.layers[0:23]):
    l1.set_weights(l2.get_weights())

print("Set untrainable")
for layer in model_Classify.layers[:23]:
    layer.trainable = False

model_Classify.load_weights("../Classify_epoch_50_batch_5.hdf5")
model_Classify.compile(optimizer=Adam(lr=1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
model_Classify.fit_generator(mygen_label.Generator(), epochs=epochs, steps_per_epoch=30, verbose=1, callbacks=classify_callbacks)

######################################################################
#
# Train classifier with all layers trainable (currently bad performance)
#
######################################################################

#print("Set trainable")
#for layer in model_Classify.layers[:23]:
#    layer.trainable = True

#classify_model_checkpoint = ModelCheckpoint("Classify_2" + ckname, monitor='loss', verbose=0, save_best_only=True, loss='binary_crossentropy')
##model_Classify.compile(optimizer=Adam(lr=1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
#classify_callbacks = [classify_model_checkpoint]
#model_Classify.load_weights("Classify_epoch_50_batch_4.hdf5");
#model_Classify.fit_generator(mygen_label.Generator(), epochs=epochs, steps_per_epoch=10, verbose=1, callbacks=classify_callbacks)

