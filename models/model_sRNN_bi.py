#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 08:18:01 2020

@author: liang
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf

from tensorflow import keras

print(tf.__version__)
print(sys.version_info)
for module in mpl, np, pd, sklearn, tf, keras:
    print(module.__name__, module.__version__)

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# load data
folder = 'data_4_model_new'


data_train = np.load(os.path.join(folder, 'train_data.npz'),'r')
data_train = data_train['arr_0']

data_valid = np.load(os.path.join(folder, 'valid_data.npz'),'r')
data_valid = data_valid['arr_0']

data_test = np.load(os.path.join(folder, 'test_data.npz'),'r')
data_test = data_test['arr_0']

print('data loaded successfully')

#%%
EPOCHS = 200
BATCH_SIZE = 50
IMG_WIDTH = 512
IMG_HEIGHT = 3

WINDOW_SIZE = 3


#%%
single_rnn_model = keras.models.Sequential([
    # 1. define matrix: [vocab_size, embedding_dim]
    # 2. [1,2,3,4..], max_length * embedding_dim
    # 3. batch_size * max_length * embedding_dim
    keras.layers.InputLayer(input_shape=(None,3)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.BatchNormalization(trainable = False),
    keras.layers.Dense(64, activation = 'relu'),
    keras.layers.Bidirectional(keras.layers.SimpleRNN(128,return_sequences = True)),
    keras.layers.Dense(64, activation = 'relu'),
    keras.layers.Dense(16, activation = 'relu'),
    keras.layers.Dense(2, activation = 'relu'),
    keras.layers.BatchNormalization(trainable = False),
    keras.layers.Dense(16, activation = 'relu'),
    keras.layers.Dense(64, activation = 'relu'),
    keras.layers.Bidirectional(keras.layers.SimpleRNN(128,return_sequences = True)),
    keras.layers.Dense(64, activation = 'relu'),
    keras.layers.BatchNormalization(trainable = False),
    keras.layers.Dense(32, activation = 'relu'),
    keras.layers.Dense(3, activation=tf.nn.leaky_relu),
])

single_rnn_model.summary()
single_rnn_model.compile(optimizer = 'adam',
                         loss = 'mean_squared_error',
                         metrics = ['mean_absolute_error'])

#%%

logdir = 'new_model/toy-sRNN-bi'
if not os.path.exists(logdir):
    os.mkdir(logdir)
#%%
output_model_file = os.path.join(logdir,
                                 "sRNN_bi_model.h5")

callbacks = [
    keras.callbacks.TensorBoard(logdir),
    keras.callbacks.ModelCheckpoint(output_model_file,
                                    save_best_only = True),
    keras.callbacks.EarlyStopping(patience=5, min_delta=5*1e-2),
]

history = single_rnn_model.fit(x = data_train, y = data_train,
                               batch_size= BATCH_SIZE,
                               validation_data = (data_valid, data_valid),
                               validation_freq = 1,
                               epochs = EPOCHS,
                               callbacks = callbacks)

#%%
def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 0.7)
    plt.show()

plot_learning_curves(history)
    