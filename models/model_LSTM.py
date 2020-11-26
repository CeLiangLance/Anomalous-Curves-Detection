# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 11:50:38 2020

@author: CeLiang
"""


#%%
import tensorflow as tf
from tensorflow import keras

print(tf.__version__)


print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))



#%%
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

#%%
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
# normalization
'''

min_normal = np.min([np.min(data_test), np.min(data_train), np.min(data_valid)])
max_normal = np.max([np.max(data_test), np.max(data_train), np.max(data_valid)])


data_train = (data_train - min_normal) / (max_normal - min_normal)
data_valid = (data_valid - min_normal) / (max_normal - min_normal)
data_test = (data_test - min_normal) / (max_normal - min_normal)

print('normalization finnished')
'''
#%%
''''

path_nor = os.path.join(folder,'normalization.npz')
np.savez(path_nor, maxi = max_normal, mini = min_normal)
'''
#%%
'''
test = np.load(path_nor,'r')
test['maxi']
'''

#%%
'''
sample_reshape = data_test[0:5,:,:]
reshaped = np.reshape(sample_reshape, (-1,3))
reshaped2 = np.reshape(sample_reshape, (-1,6))
'''

#%%
EPOCHS = 200
BATCH_SIZE = 50
IMG_WIDTH = 512
IMG_HEIGHT = 3

WINDOW_SIZE = 3
'''
data_train = np.reshape(data_train,(-1,3*WINDOW_SIZE)) 

data_valid = np.reshape(data_valid,(-1,3*WINDOW_SIZE)) 

data_test = np.reshape(data_test,(-1,3*WINDOW_SIZE)) 
print('reshaping finished')
'''

#%%







#%%
single_rnn_model = keras.models.Sequential([
    # 1. define matrix: [vocab_size, embedding_dim]
    # 2. [1,2,3,4..], max_length * embedding_dim
    # 3. batch_size * max_length * embedding_dim
    keras.layers.InputLayer(input_shape=(None,3)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.BatchNormalization(trainable = False),
    keras.layers.Dense(64, activation = 'relu'),
    keras.layers.LSTM(128,return_sequences = True),
    keras.layers.Dense(64, activation = 'relu'),
    keras.layers.BatchNormalization(trainable = False),
    keras.layers.LSTM(128,return_sequences = True),
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

logdir = 'new_model/toy-LSTM'
if not os.path.exists(logdir):
    os.mkdir(logdir)
#%%
output_model_file = os.path.join(logdir,
                                 "LSTM_model.h5")

callbacks = [
    keras.callbacks.TensorBoard(logdir),
    keras.callbacks.ModelCheckpoint(output_model_file,
                                    save_best_only = True),
    keras.callbacks.EarlyStopping(patience=5, min_delta=1e-4),
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
    plt.gca().set_ylim(0, 0.5)
    plt.show()

plot_learning_curves(history)
    
    