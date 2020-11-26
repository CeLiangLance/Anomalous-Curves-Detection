# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 08:22:15 2020

@author: CeLiang
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

clf = IsolationForest( behaviour = "new", max_samples=200, random_state = 1, contamination= 0.15)
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
visualize_streamline_removed(fiber_sub,IF_score,control_par=1)

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

#%%
name_list = np.load(os.path.join('..','data_4_model_new', 'filenames.npy'),'r')
test_list = np.load(os.path.join('..','data_4_model_new', 'valid_list.npy'),'r')

mask_score_dict = np.load('mask_score_dict.npy',allow_pickle=True).item()

cci_dict = np.load('cci_dict.npy',allow_pickle=True).item()
ALLscore_dict = np.load('ALLscore_dict_17.npy',allow_pickle=True).item()


name = test_list[0]



test_data = np.load(os.path.join('..','cci_clean_data', name),'r')
data_test = test_data['arr_0']
#%%

cci = cci_dict[os.path.splitext(name)[0]]
cci = cci_dict[os.path.splitext(name_list[0])[0]]
for i in range(1,len(name_list)):
    tmp = cci_dict[os.path.splitext(name_list[i])[0]]
    cci = np.concatenate((cci,tmp))

mask = mask_score_dict[os.path.splitext(name_list[0])[0]]
for i in range(1,len(name_list)):
    tmp = mask_score_dict[os.path.splitext(name_list[i])[0]]
    mask = np.concatenate((mask,tmp))
    

#%%
clf = IsolationForest( behaviour = "new", max_samples=200, random_state = 1, contamination= 0.10)
preds = clf.fit_predict(cov_all)
#%%
cci_thr = 1*(cci>3)
mask_thr = 1*(mask<630)
ifscore = 1*(preds>0)


print(sum(cci_thr*mask_thr*ifscore) / len(cci_thr))
print(sum(1*(cci_thr+mask_thr+ifscore)>2) / len(cci_thr))

#%%%

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
from visualize_score import  visualize_streamline ,visualize_streamline_removed
cci_0 = 1*(cci_dict[os.path.splitext(name)[0]]>3)
mask_0 = 1*(np.array(mask_score_dict[os.path.splitext(name)[0]])<630)
ifscore_0 = 1*(np.array(IFscore_dict[os.path.splitext(name)[0]])>0)
print(sum(cci_0) / len(cci_0))
print(sum(mask_0)/ len(cci_0))
print(sum(ifscore_0)/ len(cci_0))
print(sum(cci_0*ifscore_0) / len(cci_0))
print(sum(cci_0*ifscore_0*mask_0) / len(cci_0))

#%%
label = cci_0 * mask_0 * ifscore_0
visualize_streamline(data_test,label,control_par=1)
#%%
label_1 = cci_0 * mask_0 
visualize_streamline_removed(data_test,ifscore_0,control_par=1)
#%%
np.save('ALLscore_dict_10.npy', IFscore_dict) 


