# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 22:05:39 2020

@author: CeLiang

clean the raw data to fit the requirment of calaculating cci 
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
name_list = np.load(os.path.join('..','data_4_model', 'filenames.npy'),'r')

#%%

#tmp = np.unique(data_evl, axis=0)
def zero_remove(darray):
    
    for i in range(np.shape(darray)[0]-1,-1,-1):
        if not (darray[i] == 0 ).all():
            return darray[0:i+1] 
            break

def remove_indentical(n, streamlines_ori):
    
    for i in range(n):
        
        error_index = []
        subsamp_sls = set_number_of_points(streamlines_ori, 64)
        mdf1 = np.zeros((1,len(subsamp_sls)))
        for i, sl in enumerate(subsamp_sls):
            mdf_mx = bundles_distances_mdf([subsamp_sls[i]], subsamp_sls)
            mdf1 = np.vstack((mdf1,mdf_mx))
            if (mdf_mx == 0).sum() > 1:
                if i-1 not in error_index:
                    error_index.append(i)
        
        if error_index == []:
            return streamlines_ori
            print('number of detecting iteration is '+str(2*i+1))
            break
        
        streamlines_evl_final1 = Streamlines()
        for i, sl in enumerate(streamlines_ori):
            if i not in error_index:
                streamlines_evl_final1.append(sl)
                
                
        error_index = []
        subsamp_sls = set_number_of_points(streamlines_evl_final1, 64)
        mdf1 = np.zeros((1,len(subsamp_sls)))
        for i, sl in enumerate(subsamp_sls):
            mdf_mx = bundles_distances_mdf([subsamp_sls[i]], subsamp_sls)
            mdf1 = np.vstack((mdf1,mdf_mx))
            if (mdf_mx == 0).sum() > 1:
                if i-1 not in error_index:
                    error_index.append(i)
        if error_index == []:
            return streamlines_evl_final1
            print('number of detecting iteration is '+str(2*i+2))
            break
        
        streamlines_ori = Streamlines()
        for i, sl in enumerate(streamlines_evl_final1):
            if i not in error_index:
                streamlines_ori.append(sl)


#%%
'''
a = np.array([1,2,3,4,5,6])
book_dict = {}
book_dict.update({"country": a})
'''

def gen_array(strline):
    longth = len(strline)
    result = np.zeros((longth,550,3))
    for i in range(longth):
        result[i,0:len(strline[i]),:] = strline[i]
    return result


def clean_subject(ndata):
    cci = []  
    streamlines_evl = Streamlines()
    for j in range(np.shape(ndata)[0]):
        tmp = ndata[j]
        tmp = zero_remove(tmp)
        streamlines_evl.append(tmp)
        
    lengths = list(utils.length(streamlines_evl))
    long_streamlines_evl = Streamlines()
    for i, sl in enumerate(streamlines_evl):
        if lengths[i] > 40:
            long_streamlines_evl.append(sl)  
        
    streamlines_test = remove_indentical(10,long_streamlines_evl)
    cci_idv = cluster_confidence(streamlines_test,subsample=64)
    cci.append(cci_idv)
    return gen_array(streamlines_test), np.array(cci[0])


def gen_clean_data(name_list):
    cci_dict = {}
    i = 0
    for name in name_list:
    
        path = os.path.join('..','np_data',name)
        sample_data = np.load(path)
        ndata = sample_data['arr_0']
        data, cci = clean_subject(ndata)
        name_save = os.path.join('..','cci_clean_data',os.path.splitext(name)[0])
        #np.savez_compressed(name_save,data)
        cci_dict.update({os.path.splitext(name)[0]: cci})
        print('the '+str(i+1)+' finished')
        i+=1
    np.save('cci_dict.npy', cci_dict) 
       
gen_clean_data(name_list[0:5])

'''
Read the dict
tt = np.load('test.npy',allow_pickle=True).item()
ttt = tt['128127']
'''

#%%
def cal_cci(name_list,text):
    cci = []
    for i in range(len(name_list)):
        path = os.path.join('np_data',name_list[i])
        sample_data = np.load(path)
        ndata = sample_data['arr_0']
        
        streamlines_evl = Streamlines()
        for j in range(np.shape(ndata)[0]):
            tmp = ndata[j]
            tmp = zero_remove(tmp)
            streamlines_evl.append(tmp)
        streamlines_test = remove_indentical(10,streamlines_evl)
        cci_idv = cluster_confidence(streamlines_test,subsample=64)
        cci.append(cci_idv)
        print('the '+str(i+1)+' finished')
    path_save = os.path.join('data_4_model',text)
    np.save(path_save, cci)
    return cci
        
'''
        lengths = list(utils.length(streamlines_evl))
        long_streamlines_evl = Streamlines()
        for i, sl in enumerate(streamlines_evl):
            if lengths[i] > 40:
                long_streamlines_evl.append(sl)
'''
        
#%%
import time
start = time.time()
test_list = np.load(os.path.join('data_4_model', 'test_list.npy'),'r')
cci = cal_cci(test_list,'test_cci')
end = time.time()
print (end-start)