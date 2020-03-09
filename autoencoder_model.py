import numpy as np 
import os
#import skimage.io as io
#import skimage.transform as trans
from datetime import datetime
import numpy as np
import tensorflow as tf 
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from tensorflow.keras.utils import plot_model
#from ktf import backend as keras   .add(lambda(lambda x: x ** 2))

def downsample(n_filers, kernel=3, apply_batch=True):
    initializer = tf.random_normal_initializer(0,0.02)
    result = Sequential()
    result.add(Conv2D(n_filers,
                    kernel,
                    strides = 2,
                    padding = 'same',
                    kernel_initializer = initializer,
                    use_bias = not apply_batch,))
    if apply_batch:
        result.add(BatchNormalization())
    result.add(ReLU())
    return result

def upsample(n_filers, kernel = 3, apply_dropout=False):
    initializer = tf.random_normal_initializer(0,0.02)
    result = Sequential()
    result.add(Conv2DTranspose(n_filers,
                            kernel,
                            strides = 2,
                            padding = 'same',
                            kernel_initializer = initializer,
                            use_bias = False,))
    result.add(BatchNormalization())
    if apply_dropout:
        result.add(Dropout(0.5))
    result.add(ReLU())
    return result

def Segmenter(pretrained_weights = None,input_size = (256,256,3)):
    inputs = Input(input_size)
    down_stack=[
              #Lambda(lambda x: x/127.5-1),           #Normaliza los valores de los pixeles
              downsample(64,apply_batch=False),  #128,
              downsample(128),                   #64
              downsample(256),                   #32
              downsample(512),                   #16
              downsample(512),                   #8
              downsample(512),                   #4
              downsample(512),                   #2
              downsample(512),                   #1
    ]
    up_stack=[
              upsample(512,apply_dropout=True), #2,
              upsample(512,apply_dropout=True), #4
              upsample(512,apply_dropout=True), #8
              upsample(512),                   #16
              upsample(256),                   #32
              upsample(128),                   #64
              upsample(64),                   #128
    ]
    initializer = tf.random_normal_initializer(0,0.02)
    last = Conv2DTranspose(filters =1,
                         kernel_size = 3,
                         strides = 2,
                         padding='same',
                         kernel_initializer=initializer,
                         activation = 'sigmoid',)
    x = inputs
    s =[]
    concat = Concatenate()
    # CODIFICADOR
    for down in down_stack:
        x = down(x)
        s.append(x)
    s = s[::-1][1:]
    # DECODIFICADOR
    for up,sk in zip(up_stack,s):
        x = up(x)
        x = concat([x,sk])
    # Capa final de binarización
    last =  last(x)
    # Generando el modelo
    model = Model(inputs=inputs, outputs = last)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    # load pretrained weights
    if(pretrained_weights):
        model.load_weights(pretrained_weights)
    return model