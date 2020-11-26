# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 07:14:27 2020

@author: CeLiang
"""

import  numpy as np
import os

from visualize_score import  visualize_streamline , visualize_streamline_removed,visualize_streamline_true


manual_label_dict = np.load('manual_label_dict.npy',allow_pickle = True).item()
name_list = np.load(os.path.join('..','data_4_model_new', 'filenames.npy'),'r')
#%%
name = name_list[0]
test_data = np.load(os.path.join('..','cci_clean_data',name))
data_evl = test_data['arr_0']
key = os.path.splitext(name)[0]

label = manual_label_dict[key]
#visualize_streamline_true(data_evl,label) 
visualize_streamline(data_evl,label) 


#visualize_streamline_removed(data_evl,label,removed=1)


#%%
name = '155635_cleaned.npz'
test_data = np.load(os.path.join('..','ply_processing',name))
data_evl = test_data['arr_0']
visualize_streamline(data_evl,np.ones(np.shape(data_evl)[0]))


#%%

def save_render(name):
    
    test_data = np.load(os.path.join('..','cci_clean_data',name))
    data_evl = test_data['arr_0']
    key = os.path.splitext(name)[0]
    label = manual_label_dict[key]
    
    visualize_streamline(data_evl,label,save_name =key+'.png',save_able=True) 


#%%

save_render(name_list[10])

#%%
save_render(name_list[20])
save_render(name_list[30])
save_render(name_list[40])
save_render(name_list[50])















