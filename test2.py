# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 11:59:55 2020

@author: CeLiang
"""

from plyfile import PlyData
import numpy as np
import os
import tensorflow as tf

from sklearn.metrics import mean_squared_error
from dipy.tracking.streamline import Streamlines
from dipy.tracking.streamlinespeed import length
                                          

class PlyStruct:
    """Class that process poly data"""

    def __init__(self):
        self.ply_data = None
        self.idx = None
        self.properties = None
        self.stream_line = None

    def load_poly_data(self, subpath: str, num_prop: int = None):
        """
        Load ply file
        :param str subpath: path of ply file
        :param int num_prop: optional parmeter to only take the first num properties (e.g. only x,y,z coordinates)
        """
        ply = PlyData.read(subpath)
        property_list = [ply['vertices'].data[name] for name in ply['vertices'].data.dtype.names]
        self.ply_data = np.array(property_list).T
        if num_prop:
            self.ply_data = self.ply_data[:, :num_prop]
        self.idx = ply['fiber'].data['endindex']
        self.properties = ply['vertices'].data.dtype.names
        if num_prop:
            self.properties = self.properties[:num_prop]
   

    def gen_stream_line(self):
        self.stream_line = []
        self.stream_line.append(self.ply_data[0:self.idx[0],:])
        for i in range(len(self.idx)-1):
            self.stream_line.append(self.ply_data[self.idx[i]:self.idx[i+1],:])


def zero_remove(darray):
    
    for i in range(np.shape(darray)[0]-1,-1,-1):
        if not (np.around(darray[i], decimals=0) == 0 ).all():
            return darray[0:i+1] 
            break


def get_length(stream):
    # to get the length of streams in this subject
    len_table = []
    for i in range(len(stream)):
        len_table.append(np.shape(stream[i])[0])
    return len_table

def cal_mse(data_raw,data_pred):
    mse = np.zeros(np.shape(data_raw)[0])
    for i in range(np.shape(data_raw)[0]):
        result = mean_squared_error(data_raw[i], data_pred[i])
        mse[i] = result
    return mse  


def ply2np(name):
    # convert a ply format to the matrix we are using
    # input : '128127_ex_cc-body_shore.ply', name of a oly file
    # optput: a matrix (#_of_fibers, #_of_vertex, 3 )
    temo = PlyStruct()
    temo.load_poly_data(os.path.join('data',name),num_prop=3)
    #data = temo.ply_data
  
    temo.gen_stream_line()
    stream = temo.stream_line


    stream = np.array(stream)
    len_table = get_length(stream)
    subject = np.zeros((len(stream),np.max(len_table),3))
    for i in range(len(stream)):
        length = len_table[i]
        subject[i,0:length,:] = stream[i]
        
    return subject

#%%
model_list = ['deep_bi_GRU.h5', 'deep_bi_LSTM.h5', 'deep_GRU.h5', 'deep_LSTM.h5', 
              'GRU_bi_model.h5', 'GRU_dr_model.h5', 'GRU_model.h5', 'LSTM_bi_model.h5', 
              'LSTM_dr_model.h5', 'LSTM_model.h5', 'simple_rnn_2_model.h5', 
              'sRNN_bi_model.h5', 'toy_rnn_2_dr_model.h5']

#test= os.listdir('trained_models')
#%%
name = '156031_ex_cc-body_shore.ply'
bundle = ply2np(name)


model_path = os.path.join('trained_models','deep_GRU.h5')

model = tf.keras.models.load_model(model_path,custom_objects={'leaky_relu': tf.nn.leaky_relu})


denoise = model(bundle)
denoise = denoise.numpy()
mse_idv = cal_mse(bundle, denoise)


#%%
thre = np.percentile(mse_idv,85)
pred_1 = 1*(mse_idv<thre)


bundle_str = Streamlines()

for i in range(np.shape(bundle)[0]):
    tmp = bundle[i]
    tmp = zero_remove(tmp)
    #tmp = tmp[~np.all(tmp == 0, axis=-1)]
    #tmp = np.around(tmp, decimals=0)
    bundle_str.append(tmp)
    
lengths = length(bundle_str)
len_thre = 40
pred_2 = 1*(lengths > len_thre)

pred = 1*((pred_1 + pred_2)>1)



import sys
sys.path.append(r'toolkit')
from visualize_score import  visualize_streamline, visualize_streamline_removed
visualize_streamline(bundle,pred,control_par=1,)



op_path = 'result'
if not os.path.exists(op_path):
    os.mkdir(op_path)

save_path = os.path.join(op_path,'Detection'+name.split('_')[0])
np.save(save_path,pred)


#%%


from dipy.tracking.streamline import Streamlines,cluster_confidence
from dipy.tracking.distances import bundles_distances_mdf
from dipy.tracking.streamlinespeed import (compress_streamlines, length,
                                           set_number_of_points)


data_new = np.delete(bundle, np.where(pred == 0), axis=0)





from dipy.viz import colormap
from dipy.viz import actor, window

subsamp_sls = Streamlines()

for i in range(np.shape(data_new)[0]):
    tmp = data_new[i]
    tmp = zero_remove(tmp)
    #tmp = tmp[~np.all(tmp == 0, axis=-1)]
    #tmp = np.around(tmp, decimals=0)
    subsamp_sls.append(tmp)
    


color = colormap.line_colors(subsamp_sls)

streamlines_actor = actor.line(subsamp_sls,
                               colormap.line_colors(subsamp_sls))

# Create the 3D display.
scene = window.Scene()
scene.add(streamlines_actor)

window.show(scene)

#%%







t1 = np.load(os.path.join('result', 'Detection156031.npy'),allow_pickle=True)
t2 = np.load(os.path.join('result', 'Detection156031_m0.npy'),allow_pickle=True)
t3 = np.load(os.path.join('result', 'Detection156031_m2.npy'),allow_pickle=True)




print(np.sum(1*(t1==t2)))









