# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 01:51:10 2020

@author: CeLiang

generate dictinoaries to relate subject name with score
"""
import  numpy as np
import os
from sklearn import preprocessing
from dipy.tracking.streamlinespeed import  length
from dipy.tracking.streamline import Streamlines  
#%%

#mask_scores = np.load('score_list.npy')
name_list = np.load(os.path.join('..','data_4_model_new', 'filenames.npy'),'r')
folder = 'data_4_model_new'
folder_np = 'cci_clean_data'
cci_dict = np.load('cci_dict.npy',allow_pickle=True).item()
mask_score_dict = np.load('mask_score_dict.npy',allow_pickle=True).item()

#%%

def zero_remove(darray):
    
    for i in range(np.shape(darray)[0]-1,-1,-1):
        if not (np.around(darray[i], decimals=0) == 0 ).all():
            return darray[0:i+1] 
            break

def gen_len_list(name_list):
    
    len_list = [0]
    
    for i in range(len(name_list)):
        
        name = name_list[i]
        test_data = np.load(os.path.join('..',folder_np, name),'r')
        test_data = test_data['arr_0']
        num = np.shape(test_data)[0]
        len_list.append( num + len_list[-1])
    return len_list

len_list = gen_len_list(name_list)

#%%

def cal_cov_sub(sub_name):
    
    fiber_sub = np.load(os.path.join('..','cci_clean_data',sub_name))
    fiber_sub = fiber_sub['arr_0']
    cci = cci_dict[os.path.splitext(sub_name)[0]]
    mask_score = mask_score_dict[os.path.splitext(sub_name)[0]]

    streamlines_evl = Streamlines()
    
    for i in range(np.shape(fiber_sub)[0]):
        tmp = fiber_sub[i]
        tmp = zero_remove(tmp)
        streamlines_evl.append(tmp)
        
    lengths = np.array(list(length(streamlines_evl)))
    
    #==============
    fiber_one = fiber_sub[0].transpose()
    fiber_one_std = preprocessing.scale(fiber_one)
    
    covarience = np.cov(fiber_one_std)
    result = np.array( [mask_score[0],cci[0],lengths[0],covarience[0,0], covarience[1,1], covarience[2,2],covarience[0,1], covarience[0,2],covarience[1,2]]).transpose()
    
    for i in range(1,np.shape(fiber_sub)[0]):
            fiber_one = fiber_sub[i].transpose()
            fiber_one_std = preprocessing.scale(fiber_one)
            
            covarience = np.cov(fiber_one_std)
            #tmp = np.array( [covarience[0,0], covarience[1,1], covarience[2,2]]).transpose()
            tmp = np.array( [mask_score[i],cci[i],lengths[i],covarience[0,0], covarience[1,1], covarience[2,2],covarience[0,1], covarience[0,2],covarience[1,2]]).transpose()
            result = np.vstack((result,tmp))
    return result


def cal_cov_whole(name_list):
    sub_name = name_list[0]
    result = cal_cov_sub(sub_name)
    #cov_dict = {}
    #cov_dict.update({os.path.splitext(sub_name)[0]: result})
    
    for i in range(1,len(name_list)):
        sub_name = name_list[i]
        tmp = cal_cov_sub(sub_name)
        result = np.vstack((result,tmp))
        #cov_dict.update({os.path.splitext(sub_name)[0]: tmp})
        #print('the'+str(i+1)+'finished')
    
    
    return result    

#test_result, cov_dict = cal_cov_whole(name_list)
   
#%%
    
cov_all = cal_cov_whole(name_list)
#%%%
#np.save(os.path.join('toolkit','cov_all'),test_result)
#test_result = np.load(os.path.join('cov_all.npy'))


from sklearn.ensemble import IsolationForest
np.random.seed(1)

clf = IsolationForest( behaviour = "new", max_samples=200, random_state = 1, contamination= 0.17)
preds = clf.fit_predict(cov_all)



#%%

from visualize_score import  visualize_streamline  

name = name_list[0]
key = os.path.splitext(name)[0]

IF_score = preds[0:len_list[1]]
IF_score = np.array(IF_score)
IF_score = IF_score +1


fiber_sub = np.load(os.path.join('..','cci_clean_data',name))
fiber_sub = fiber_sub['arr_0']



visualize_streamline(fiber_sub,IF_score,control_par=1)


#%%


def IFscore_2_dict(preds, len_list, name_list):
    IFscore_dict = {}
    for i in range(len(name_list)):
        name = name_list[i]
        start = len_list[i]
        end = len_list[i+1]
        IFscore_dict.update({os.path.splitext(name)[0]: preds[start:end]})
    return IFscore_dict

IFscore_dict = IFscore_2_dict(preds, len_list, name_list)

#%%

np.save('ALLscore_dict_17.npy', IFscore_dict) 

#%%


name_list = np.load(os.path.join(folder, 'filenames.npy'),'r')

avg_dist = np.load(os.path.join('toolkit','avg_dist.npy'))

ref_affine = np.array([[  -1.25,    0.  ,    0.  ,   90.  ],
                       [   0.  ,    1.25,    0.  , -126.  ],
                       [   0.  ,    0.  ,    1.25,  -72.  ],
                       [   0.  ,    0.  ,    0.  ,    1.  ]])




#%%
def cal_dist_score(dist_array,position_array):
    # per fiber
    result = 0
    for i in range(np.shape(position_array)[0]):
        x,y,z = position_array[i]
        tmp = dist_array[x,y,z]
        result += tmp
    return result

#dist_score = cal_dist_score(avg_dist,cc_uniques)


#%%
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
        
    return scores


#scores = cal_score_subject(test_data)
    
#%%

def gen_score_dict(name_list):
    
    name = name_list[0]
    test_data = np.load(os.path.join(folder_np, name),'r')
    test_data = test_data['arr_0']
    scores = cal_score_subject(test_data)
    mask_score_dict = {}
    mask_score_dict.update({os.path.splitext(name)[0]: scores})
    
    
    for i in range(1,np.shape(name_list)[0]):        
        name = name_list[i]
        test_data = np.load(os.path.join(folder_np, name),'r')
        test_data = test_data['arr_0']
        scores_tmp = cal_score_subject(test_data)
        mask_score_dict.update({os.path.splitext(name)[0]: scores_tmp})
        #scores = np.concatenate((scores, scores_tmp),axis = 0)
    
    np.save('mask_score_dict.npy', mask_score_dict) 
    return mask_score_dict


#mask_score_dict = gen_score_dict(name_list)

#%%
mask_score_dict = np.load('mask_score_dict.npy',allow_pickle=True).item()













