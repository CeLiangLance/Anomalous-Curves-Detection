# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 06:42:29 2020

@author: CeLiang
"""
from dipy.tracking.streamline import Streamlines,cluster_confidence
import numpy as np
from dipy.tracking import utils

#from dipy.viz import actor, window

#import matplotlib.pyplot as plt
from dipy.tracking.distances import bundles_distances_mdf
from dipy.tracking.streamlinespeed import (compress_streamlines, length,
                                           set_number_of_points)
import os

#%%

def zero_remove(darray):
    
    for i in range(np.shape(darray)[0]-1,-1,-1):
        if not (darray[i] == 0 ).all():
            return darray[0:i+1] 
            break

#%%

sub_len = 50
#%%
folder = 'data_4_model_new'
name_list = np.load(os.path.join('..',folder, 'filenames.npy'),'r')

def gen_short_data(name_list):
    for i in range(len(name_list)):
        name = name_list[i]
        path = os.path.join('..','cci_clean_data',name)
        sample_data = np.load(path)
        ndata = sample_data['arr_0']
        
        streamlines_evl = Streamlines()
        for j in range(np.shape(ndata)[0]):
            tmp = ndata[j]
            tmp = zero_remove(tmp)
            streamlines_evl.append(tmp)
            
        
        subsamp_sls = set_number_of_points(streamlines_evl, sub_len)
        
        tt= np.array(subsamp_sls)
        path = os.path.join('..','subsample-data',str(sub_len))
        if not os.path.exists(path):
            os.makedirs(path)
        name_save = os.path.join(path,os.path.splitext(name)[0])
        
        np.savez_compressed(name_save,tt)
        print(str(i+1)+'/141 finished')
gen_short_data(name_list)
#%%


#%%

def npz2data(filnames, name):
    test_name = filnames[0]
    test_data = np.load(os.path.join('..',folder, test_name),'r')
    data_out = test_data['arr_0']


    for i in range(1, len(filnames)):
        filename = filnames[1]
        data = np.load(os.path.join('..',folder, filename),'r')
        tmp_out = data['arr_0']
    
        data_out = np.concatenate((data_out, tmp_out))
    

    path = os.path.join('..','short_data_4_model',str(sub_len))
    if not os.path.exists(path):
        os.makedirs(path)
        
    name_save = os.path.join(path, name+'_data')
    #list_save = os.path.join(path, name+'_list')
    np.savez_compressed(name_save,data_out)
    #np.save(list_save, filnames)
    
    
    
    
#%%
name_list = np.load(os.path.join('..','short_data_4_model','filenames.npy'))
train_list = np.load(os.path.join('..','short_data_4_model','train_list.npy'))
test_list = np.load(os.path.join('..','short_data_4_model','test_list.npy'))
valid_list = np.load(os.path.join('..','short_data_4_model','valid_list.npy'))    

folder = os.path.join('subsample-data',str(sub_len))
npz2data(train_list, 'train')
print(1)
npz2data(valid_list, 'valid')
print(2)
npz2data(test_list, 'test')
print(3)  
    



#%%
from dipy.viz import colormap
from dipy.viz import actor, window



color = colormap.line_colors(subsamp_sls)

streamlines_actor = actor.line(subsamp_sls,
                               colormap.line_colors(subsamp_sls))

# Create the 3D display.
scene = window.Scene()
scene.add(streamlines_actor)

window.show(scene)





#%%



#%%









