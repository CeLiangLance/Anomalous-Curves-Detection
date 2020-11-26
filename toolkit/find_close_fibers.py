# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 04:16:56 2020

@author: CeLiang


"""

from dipy.tracking.streamline import Streamlines
import numpy as np
from dipy.tracking import utils

from dipy.viz import actor, window

import matplotlib.pyplot as plt

import os

from dipy.tracking.distances import bundles_distances_mdf
from dipy.tracking.streamlinespeed import (compress_streamlines, length,
                                           set_number_of_points)
from visualize_score import  visualize_streamline , visualize_streamline_removed

#%%

cci_dict = np.load('cci_dict.npy',allow_pickle=True).item()
name_list = np.load(os.path.join('..','data_4_model_new', 'filenames.npy'),'r')
folder = 'data_4_model_new'
folder_np = 'cci_clean_data'


name = name_list[0]
test_data = np.load(os.path.join('..',folder_np, name),'r')
data_evl = test_data['arr_0']


#%%
#tmp = np.unique(data_evl, axis=0)
def zero_remove(darray):
    
    for i in range(np.shape(darray)[0]-1,-1,-1):
        if not (np.around(darray[i], decimals=0) == 0 ).all():
            return darray[0:i+1] 
            break
        

#%%
streamlines_evl = Streamlines()

for i in range(np.shape(data_evl)[0]):
    tmp = data_evl[i]
    tmp = zero_remove(tmp)
    #tmp = tmp[~np.all(tmp == 0, axis=-1)]
    #tmp = np.around(tmp, decimals=0)
    streamlines_evl.append(tmp)





#%%

subsamp_sls = set_number_of_points(streamlines_evl, 64)
mdf1 = np.zeros((1,len(subsamp_sls)))

mdf_mx = bundles_distances_mdf([subsamp_sls[0]], subsamp_sls)
'''
            mdf1 = np.vstack((mdf1,mdf_mx))
            if (mdf_mx == 0).sum() > 1:
                if i-1 not in error_index:
                    error_index.append(i)
'''


thre = np.percentile(mdf_mx, 15)
label = 1*(mdf_mx < thre).flatten()
cci_roi = cci_dict[os.path.splitext(name)[0]] * label

from visualize_score import  visualize_streamline  

#visualize_streamline(data_evl,cci_roi,control_par=1)


#%%




cci_roi = cci_dict[os.path.splitext(name)[0]] * label

data_new = np.delete(data_evl, np.where(cci_roi == 0), axis=0)

cci_roi_new = np.delete(cci_roi, np.where(cci_roi == 0), axis=0)

visualize_streamline(data_new,cci_roi_new,control_par=1)


#%%




#%%
visualize_streamline(data_test,cci_roi_test,control_par=1)


#%%



    

name = name_list[0]
test_data = np.load(os.path.join('..',folder_np, name),'r')
data_evl = test_data['arr_0']


streamlines_evl = Streamlines()
for i in range(np.shape(data_evl)[0]):
    tmp = data_evl[i]
    tmp = zero_remove(tmp)
    streamlines_evl.append(tmp)
    
lengths = np.array(list(utils.length(streamlines_evl)))

mark = np.zeros((np.shape(data_evl)[0]))

subsamp_sls = set_number_of_points(streamlines_evl, 64)
for i in range(len(streamlines_evl)):
    
    mdf_mx = bundles_distances_mdf([subsamp_sls[i]], subsamp_sls)
    
    thre = np.percentile(mdf_mx, 15)
    label = 1*(mdf_mx < thre).flatten()
    cci_roi = cci_dict[os.path.splitext(name)[0]] * label
    
    cci_roi_remove = np.delete(cci_roi, np.where(cci_roi == 0), axis=0)
    cci_thre = np.percentile(cci_roi_remove, 15)
    tmp = 1*( cci_roi > cci_thre)
    mark = mark + tmp

#%%
mark_test = 1*(mark>5)
visualize_streamline(data_evl,mark_test,control_par=1)


#%%
name = name_list[0]


def find_bad_nrigh (name, score):

   
    test_data = np.load(os.path.join('..',folder_np, name),'r')
    data_evl = test_data['arr_0']
    
    streamlines_evl = Streamlines()

    for i in range(np.shape(data_evl)[0]):
        tmp = data_evl[i]
        tmp = zero_remove(tmp)
        #tmp = tmp[~np.all(tmp == 0, axis=-1)]
        #tmp = np.around(tmp, decimals=0)
        streamlines_evl.append(tmp)
    
    neighb = np.zeros((np.shape(data_evl)[0]))
    
    subsamp_sls = set_number_of_points(streamlines_evl, 64)
    for i in range(len(streamlines_evl)):
        
        mdf_mx = bundles_distances_mdf([subsamp_sls[i]], subsamp_sls)
        if (score[i] == 0): # bad fiber
        
            thre = np.percentile(mdf_mx, 5)
            len_bad = lengths[i]
            bound = len_bad * 0.1
            len_limt = 1*((lengths>len_bad-bound) & (lengths<len_bad+bound))
            label = 1*(mdf_mx < thre).flatten() # bad neighbour
            neighb = neighb + label *len_limt
    return data_evl, neighb
            
    

#%%

neighb_test = 1*(neighb < 7 )
#visualize_streamline(data_evl,neighb_test,control_par=1)

visualize_streamline_removed(data_evl,neighb_test,control_par=1,removed=1)
#%%
aa = np.array([1,2,3,4,5,6,7,8,9])
ans = 1*((aa>2) & (aa < 7))

















