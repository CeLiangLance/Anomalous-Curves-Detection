# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 05:42:02 2020

@author: CeLiang
"""
import argparse
from plyfile import PlyData
import numpy as np
import os

from sklearn.ensemble import IsolationForest

from dipy.tracking.streamline import Streamlines, cluster_confidence

from dipy.tracking.distances import bundles_distances_mdf
from dipy.tracking.streamlinespeed import (compress_streamlines, length,
                                           set_number_of_points)
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

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
    temo.load_poly_data(os.path.join('..','data',name),num_prop=3)
    #data = temo.ply_data
  
    temo.gen_stream_line()
    stream = temo.stream_line


    stream = np.array(stream)
    len_table = get_length(stream)
    subject = np.zeros((len(stream),np.max(len_table),3))
    for i in range(len(stream)):
        lengt = len_table[i]
        subject[i,0:lengt,:] = stream[i]
        
    return subject

#%%
parser = argparse.ArgumentParser()



parser.add_argument('-f', action='store',
                    dest='file_name',
                    help='Input the file name')

parser.add_argument('-IF', action='store',
                    dest='IF_thre',
                    default = 0.17,
                    help='Specify the Isolation Forest threshold')

parser.add_argument('-dis', action='store',
                    dest='dis_thre',
                    default = 630,
                    help='Specify the distance score threshold')

parser.add_argument('-CCI', action='store',
                    dest='cci_thre',
                    default = 2,
                    help='Specify the CCI threshold')

parser.add_argument('-l', action='store',
                    dest='len_con',
                    default = 40,
                    help='Specify length under what value is anomalous')


results = parser.parse_args()

print('file_name           = {!r}'.format(results.file_name))
print('Isolation Forest threshold   = {!r}'.format(results.IF_thre))
print('CCI threshold   = {!r}'.format(results.cci_thre))
print('Distance score threshold   = {!r}'.format(results.dis_thre))
print('length  control   = {!r}'.format(results.len_con))


#%%

name = results.file_name

bundle = ply2np(name)


def cal_cov_sub(fiber_sub):
    

    streamlines_evl = Streamlines()
    
    for i in range(np.shape(fiber_sub)[0]):
        tmp = fiber_sub[i]
        tmp = zero_remove(tmp)
        streamlines_evl.append(tmp)
        

    
    #==============
    fiber_one = fiber_sub[0].transpose()
    fiber_one_std = preprocessing.scale(fiber_one)
    
    covarience = np.cov(fiber_one_std)
    result = np.array( [covarience[0,0], covarience[1,1], covarience[2,2],covarience[0,1], covarience[0,2],covarience[1,2]]).transpose()
    
    for i in range(1,np.shape(fiber_sub)[0]):
            fiber_one = fiber_sub[i].transpose()
            fiber_one_std = preprocessing.scale(fiber_one)
            
            covarience = np.cov(fiber_one_std)
            #tmp = np.array( [covarience[0,0], covarience[1,1], covarience[2,2]]).transpose()
            tmp = np.array( [covarience[0,0], covarience[1,1], covarience[2,2],covarience[0,1], covarience[0,2],covarience[1,2]]).transpose()
            result = np.vstack((result,tmp))
    return result


result_if = cal_cov_sub(bundle)
#%%

np.random.seed(1)

clf = IsolationForest( behaviour = "new", max_samples=200, random_state = 1, contamination=results.IF_thre )
pred_if = clf.fit_predict(result_if)




#%%


avg_dist = np.load(os.path.join('..','toolkit','avg_dist.npy'))

ref_affine = np.array([[  -1.25,    0.  ,    0.  ,   90.  ],
                       [   0.  ,    1.25,    0.  , -126.  ],
                       [   0.  ,    0.  ,    1.25,  -72.  ],
                       [   0.  ,    0.  ,    0.  ,    1.  ]])





def cal_dist_score(dist_array,position_array):
    # per fiber
    result = 0
    for i in range(np.shape(position_array)[0]):
        x,y,z = position_array[i]
        tmp = dist_array[x,y,z]
        result += tmp
    return result



def cal_score_subject(subject_array):
    #per subject
    scores = []
    for i in range(np.shape(subject_array)[0]):
    
        aa = subject_array[i]
        aa2 = zero_remove(aa)
        aa3 = aa2.T 
        
        bb = np.ones((4,np.shape(aa3)[1]))
        bb[0:3,:]=aa3
        cc = np.dot( np.linalg.inv(ref_affine),bb)
        
        cc_int = cc[0:3,:].T.astype(np.int16)
        cc_uniques = np.unique(cc_int,axis=0)
        
        score_tmp = cal_dist_score(avg_dist,cc_uniques)
        scores.append(score_tmp)
        
    return np.array(scores)

result_dis = cal_score_subject(bundle)
#%%



streamlines_evl = Streamlines()
for j in range(np.shape(bundle)[0]):
    tmp = bundle[j]
    tmp = zero_remove(tmp)
    streamlines_evl.append(tmp)

result_cci = cluster_confidence(streamlines_evl,subsample=64)


#%%

preds_cci = 1*(result_cci>results.cci_thre)
preds_dis  = 1*(result_dis<results.dis_thre)
pred_if = 1*(pred_if>0)

pred_1 = 1*((preds_cci + preds_dis +pred_if)>2)

bundle_str = Streamlines()

for i in range(np.shape(bundle)[0]):
    tmp = bundle[i]
    tmp = zero_remove(tmp)
    #tmp = tmp[~np.all(tmp == 0, axis=-1)]
    #tmp = np.around(tmp, decimals=0)
    bundle_str.append(tmp)
    
lengths = length(bundle_str)
len_thre = results.len_con
pred_2 = 1*(lengths > len_thre)

pred = 1*((pred_1 + pred_2)>1)

save_path = 'Detection'+name.split('_')[0]+'_manual'
np.save(save_path,pred)

#%%

#tt = np.load('Detection156031_manual.npy')