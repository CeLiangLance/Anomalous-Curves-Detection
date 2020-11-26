# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 02:09:14 2020

@author: CeLiang
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 14:17:29 2020

@author: CeLiang



visuliaze streamlines wrt score

remember to set the index of list!
"""
import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

import sklearn
import pandas as pd

import sys
import time
import tensorflow as tf

config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=8, 
                        inter_op_parallelism_threads=8, 
                        allow_soft_placement=True)

session = tf.compat.v1.Session(config=config)



from tensorflow import keras

from dipy.tracking import utils
from dipy.tracking.streamline import Streamlines,cluster_confidence
from dipy.tracking.distances import bundles_distances_mdf
from dipy.tracking.streamlinespeed import (compress_streamlines, length,
                                           set_number_of_points)

from sklearn.metrics import mean_squared_error
from dipy.viz import actor, window

def zero_remove(darray):
    
    for i in range(np.shape(darray)[0]-1,-1,-1):
        if not (np.around(darray[i], decimals=0) == 0 ).all():
            return darray[0:i+1] 
            break

def visualize_streamline(darray, score, save_able=False, save_name = 'default.png', control_par =1,hue = [0.5, 1]):
    
    data_evl = darray
     
    streamlines_evl = Streamlines()
    
    for i in range(np.shape(data_evl)[0]):
        tmp = data_evl[i]
        tmp = zero_remove(tmp)
        #tmp = tmp[~np.all(tmp == 0, axis=-1)]
        #tmp = np.around(tmp, decimals=0)
        streamlines_evl.append(tmp)
    
    
    
    mse_nor = score

    


    # Visualize the streamlines, colored by cci
    ren = window.Scene()
    

    saturation = [0.0, 1.0]
    
    lut_cmap = actor.colormap_lookup_table(scale_range=(min(mse_nor), max(mse_nor)/control_par),
                                           hue_range=hue,
                                           saturation_range=saturation)
    
    bar3 = actor.scalar_bar(lut_cmap)
    ren.add(bar3)
    
    stream_actor = actor.line(streamlines_evl, mse_nor, linewidth=0.1,
                              lookup_colormap=lut_cmap)
    ren.add(stream_actor)
    
    
    if not save_able:        
        interactive = True
        if interactive:
            window.show(ren)
    
    if save_able:    
        window.record(ren, n_frames=1, out_path=save_name,
                  size=(800, 800))


#%%
def visualize_streamline_removed (darray, score, save_able=False, save_name = 'default.png', control_par =1, removed =1):

    data_new = np.delete(darray, np.where(score == removed), axis=0)

    score_new = np.delete(score, np.where(score == removed), axis=0)
    
    visualize_streamline(data_new,score_new,control_par=1)




def visualize_streamline_true (darray, score, save_able=False, save_name = 'default.png', control_par =1, removed =0):

    data_new = np.delete(darray, np.where(score == removed), axis=0)

    score_new = np.delete(score, np.where(score == removed), axis=0)
    
    score_new[0] =0
    visualize_streamline(data_new,score_new,control_par=1)





