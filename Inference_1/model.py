import numpy as np
import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras import backend as keras
from keras.callbacks import ModelCheckpoint, LearningRateScheduler


def Conv2D_layer(inputs, num_filter, kernel_size=3, activate='relu', padding='same', kern_init='he_normal', batch_normalize=False):
    conv = Conv2D(num_filter, kernel_size, activation = activate, padding = padding, kernel_initializer = kern_init)(inputs)
    if batch_normalize:
        conv = BatchNormalization()(conv)
    conv = Conv2D(num_filter, kernel_size, activation = activate, padding = padding, kernel_initializer = kern_init)(conv)
    if batch_normalize:
        conv = BatchNormalization()(conv)
    return conv

def Conv2D_up(inputs, num_filter, kernel_size=2, activate='relu', padding='same', kern_init='he_normal', batch_normalize=False):
    up = Conv2D(num_filter, kernel_size=2, activation = activate, padding = padding, kernel_initializer = kern_init)(UpSampling2D(size = (2,2))(inputs))
    if batch_normalize:
        up = BatchNormalization()(up)
    return up

def Dense_layer(inputs, out_dim, activation='relu', kern_init='he_normal', batch_normalize=False, dropout=True, drop_ratio=0.5):
    dense = Dense(out_dim, activation=activation)(inputs)
    if batch_normalize:
        dense = BatchNormalization()(dense)
    if dropout:
        dense = Dropout(drop_ratio)(dense)
    return dense

def Encoder_Classify(input_size=(512, 512, 6), num_classes=1, pretrained_weights=None, batch_normal=False, loss='binary_crossentropy'):
    inputs = Input(input_size)

    conv1 = Conv2D_layer(inputs, 64, 3, batch_normalize=batch_normal)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    # size/2: 256 x 256 x 64

    conv2 = Conv2D_layer(pool1, 128, 3, batch_normalize=batch_normal)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    # size/4: 128 x 128 x 128

    conv3 = Conv2D_layer(pool2, 256, 3, batch_normalize=batch_normal)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # size/8: 64 x 64 x 256

    conv4 = Conv2D_layer(pool3, 512, 3, batch_normalize=batch_normal)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2,2))(drop4)
    # size/16: 32 x 32 x 512

    conv5 = Conv2D_layer(pool4, 1024, 3, batch_normalize=batch_normal)
    pool5 = MaxPooling2D(pool_size=(2,2))(conv5)
    # 16 x 16 x 1024

    conv6 = Conv2D_layer(pool5, 512, 3, batch_normalize=batch_normal)
    pool6 = MaxPooling2D(pool_size=(2,2))(conv6)
    # 8 x 8 x 512

    conv7 = Conv2D_layer(pool6, 256, 3, batch_normalize=batch_normal)
    pool7 = MaxPooling2D(pool_size=(2,2))(conv7)
    # 4 x 4 x 256

    flat6 = Flatten()(conv7)
    # 4096
    drop7 = Dense_layer(flat6, 4096, batch_normalize=batch_normal)

    drop8 = Dense_layer(drop7, 4096, batch_normalize=batch_normal)

    drop9 = Dense_layer(drop8, 1024, batch_normalize=batch_normal)

#    dense10 = Dense(num_classes, activation='softmax')(drop9)
    dense10 = Dense(num_classes, activation='sigmoid')(drop9)

    model = Model(inputs = inputs, outputs = dense10)
    model.compile(optimizer=Adam(lr=1e-4), loss = loss, metrics = ['accuracy'])
    model.summary()
    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model

def Unet(input_size=(512, 512, 6), pretrained_weights=None, batch_normal=False, loss='binary_crossentropy'):
    inputs = Input(input_size)

    conv1 = Conv2D_layer(inputs, 64, 3, batch_normalize=batch_normal)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D_layer(pool1, 128, 3, batch_normalize=batch_normal)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D_layer(pool2, 256, 3, batch_normalize=batch_normal)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D_layer(pool3, 512, 3, batch_normalize=batch_normal)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2,2))(drop4)

    conv5 = Conv2D_layer(pool4, 1024, 3, batch_normalize=batch_normal)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D_up(drop5, 512, 2, batch_normalize=batch_normal)

    merge6 = concatenate([drop4, up6], axis = 3)
    conv6 = Conv2D_layer(merge6, 512, 3, batch_normalize=batch_normal)

    up7 = Conv2D_up(conv6, 256, 2, batch_normalize=batch_normal)

    merge7 = concatenate([conv3, up7], axis = 3)
    conv7 = Conv2D_layer(merge7, 256, 3, batch_normalize=batch_normal)

    up8 = Conv2D_up(conv7, 128, 2, batch_normalize=batch_normal)

    merge8 = concatenate([conv2, up8], axis = 3)
    conv8 = Conv2D_layer(merge8, 128, 3, batch_normalize=batch_normal)

    up9 = Conv2D_up(conv8, 64, 2, batch_normalize=batch_normal)

    merge9 = concatenate([conv1, up9], axis = 3)
    conv9 = Conv2D_layer(merge9, 64, 3, batch_normalize=batch_normal)

    #conv9 = Conv2D(6, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    #conv9 = Conv2D(6, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    #conv10 = Conv2D(6, 1, activation = 'sigmoid')(conv9)
    conv10 = Conv2D(6, 3, activation = 'sigmoid', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    model = Model(inputs = inputs, outputs = conv10)
    model.compile(optimizer=Adam(lr=1e-4), loss = loss,  metrics = ['accuracy'])
    model.summary()
    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model
