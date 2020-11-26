# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 03:06:19 2020

@author: CeLiang

"""


import  numpy as np
import os

from visualize_score import  visualize_streamline , visualize_streamline_removed
from dipy.tracking.streamline import Streamlines

from dipy.tracking.streamlinespeed import (compress_streamlines, length,
                                           set_number_of_points)
from dipy.tracking.distances import bundles_distances_mdf

#%%
name_list = np.load(os.path.join('..','data_4_model_new', 'filenames.npy'),'r')

mask_score_dict = np.load('mask_score_dict.npy',allow_pickle=True).item()
IFscore_dict = np.load('IFscore_dict_17.npy',allow_pickle=True).item()
cci_dict = np.load('cci_dict.npy',allow_pickle=True).item()
ALLscore_dict = np.load('ALLscore_dict_17.npy',allow_pickle=True).item()

#%%

def zero_remove(darray):
    
    for i in range(np.shape(darray)[0]-1,-1,-1):
        if not (np.around(darray[i], decimals=0) == 0 ).all():
            return darray[0:i+1] 
            break

def visualize_cleaned_result(name):
    key = os.path.splitext(name)[0]
    mask_score = mask_score_dict[key]
    mask_score = np.array(mask_score)
    IF_score = ALLscore_dict[key]
    IF_score = np.array(IF_score)
    IF_score = IF_score +1
    mask_score = 2*(mask_score<570)
    final = IF_score + mask_score
    fiber_sub = np.load(os.path.join('..','cci_clean_data',name))
    fiber_sub = fiber_sub['arr_0']
    visualize_streamline(fiber_sub,final,control_par=1)
    
#%%
#visualize_cleaned_result(name_list[0])



#%%


def find_outlier_org(name, thre):
    # thre: 0 absolute outlier
    #       1 high prob outlier
    key = os.path.splitext(name)[0]
    mask_score = mask_score_dict[key]
    mask_score = np.array(mask_score)
    IF_score = IFscore_dict[key]
    cci_score = cci_dict[key]
    cci_score = np.array(cci_score)
    IF_score = np.array(IF_score)
    IF_score = IF_score +1
    ALL_score = ALLscore_dict[key]
    ALL_score = np.array(ALL_score)
    ALL_score = ALL_score +1
    
    fiber_sub = np.load(os.path.join('..','cci_clean_data',name))
    fiber_sub = fiber_sub['arr_0']
    
    mask_score_test = 1*(mask_score<570)
    final = IF_score + mask_score_test
    cci_score_test = 1* (cci_score>2)
    
    ALL_score_test = 1*(ALL_score>0)
    
    final = cci_score_test + mask_score_test + ALL_score_test
    final = 1* (final >thre)
    return fiber_sub, final


def find_outlier(name, thre):
    # thre: 0 absolute outlier
    #       1 high prob outlier
    key = os.path.splitext(name)[0]
    mask_score = mask_score_dict[key]
    mask_score = np.array(mask_score)
    IF_score = IFscore_dict[key]
    cci_score = cci_dict[key]
    cci_score = np.array(cci_score)
    IF_score = np.array(IF_score)
    IF_score = IF_score +1
    ALL_score = ALLscore_dict[key]
    ALL_score = np.array(ALL_score)
    ALL_score = ALL_score +1
    
    fiber_sub = np.load(os.path.join('..','cci_clean_data',name))
    fiber_sub = fiber_sub['arr_0']
    
    mask_score_test = 1*(mask_score<650)
    final = IF_score + mask_score_test
    cci_score_test = 1* (cci_score>8)
    
    ALL_score_test = 1*(ALL_score>0)
    
    final = cci_score_test + mask_score_test + ALL_score_test
    final = 1* (final >thre)
    return fiber_sub, final


#visualize_streamline(fiber_sub,final,control_par=1)

def find_bad_nrigh (name, score):

   
    test_data = np.load(os.path.join('..','cci_clean_data',name))
    data_evl = test_data['arr_0']
    key = os.path.splitext(name)[0]
    cci_score = cci_dict[key]
    cci_score = np.array(cci_score)
    
    
    streamlines_evl = Streamlines()

    for i in range(np.shape(data_evl)[0]):
        tmp = data_evl[i]
        tmp = zero_remove(tmp)
        #tmp = tmp[~np.all(tmp == 0, axis=-1)]
        #tmp = np.around(tmp, decimals=0)
        streamlines_evl.append(tmp)
        
    lengths = np.array(list(length(streamlines_evl)))
    
    neighb = np.zeros((np.shape(data_evl)[0]))
    
    subsamp_sls = set_number_of_points(streamlines_evl, 64)
    for i in range(len(streamlines_evl)):
        
        mdf_mx = bundles_distances_mdf([subsamp_sls[i]], subsamp_sls)
        if (score[i] == 0): # bad fiber
        
            thre = np.percentile(mdf_mx, 4)
            len_bad = lengths[i]
            bound = len_bad * 0.1
            len_limt = 1*((lengths>len_bad-bound) & (lengths<len_bad+bound))
            label = 1*(mdf_mx < thre).flatten() # bad neighbour
            neighb = neighb + label *(cci_score<10)
    return data_evl, neighb

#%%
'''
name = name_list[0]

fiber_sub, outlier = find_outlier(name,1)
_, outlier_abs = find_outlier(name,0)
_, outlier_org = find_outlier_org(name,1)
_, outlier_neigh = find_bad_nrigh(name,outlier_abs)

#%%
fiber_sub, outlier_over = find_outlier(name,2)
_, outlier_org_over = find_outlier_org(name,2)

_, outlier_neigh_over = find_bad_nrigh(name,outlier_over)
#%%
aa = 1*(outlier_neigh_over<22)

visualize_streamline(fiber_sub,aa)
#%%
bb = 1*((aa+ outlier) >1)
visualize_streamline(fiber_sub,bb)
#%%
visualize_streamline_removed(fiber_sub,bb,removed=1)

#%%
#fiber_sub, final = final_outlier(name)
#%%
_, outlier_org_over = find_outlier_org(name,2)
print(sum(outlier_org_over))
find_overlap(final, outlier_org_over)


print( sum(final*outlier_org_over)/(sum(a)+sum(b)))

#%%
visualize_streamline(fiber_sub,outlier_over)


#%%

visualize_streamline_removed(fiber_sub,outlier,removed=1)

find_overlap(outlier, outlier_org)


#%%
'''
def final_outlier(name):
    
    fiber_sub, outlier = find_outlier(name,1)
    _, outlier_abs = find_outlier(name,0)
    #_, outlier_abs = find_outlier_org(name,0)
    _, outlier_org = find_outlier_org(name,1)
    _, outlier_neigh = find_bad_nrigh(name,outlier_abs)
    
    
    _, outlier_over = find_outlier(name,2)
    #_, outlier_org_over = find_outlier_org(name,2)
    
    _, outlier_neigh_over = find_bad_nrigh(name,outlier_over)
    aa = 1*(outlier_neigh_over<22)
    
    outlier_neigh_test = 1*(outlier_neigh<=1 ) 
    final = (1-outlier_neigh_test) + (1- outlier_org)
    final = 1- 1*(final >0)
    
    bb = 1*((aa+ final) >1)
    
    return fiber_sub, bb

#%%

def gen_manual_label(name_list):
    manual_label_dict = {}
    for i in range(len(name_list)):
        name = name_list[i]
        _, final = final_outlier(name)
        manual_label_dict.update({os.path.splitext(name)[0]: final})
        print('the ' +str(i)+'/144 finished')
        
    np.save('manual_label_dict.npy', manual_label_dict) 
    return manual_label_dict        
        

gen_manual_label(name_list)


    
#%%








#%%
def final_outlier_modi(name):
    
    fiber_sub, outlier_abs = find_outlier(name,0)
    #_, outlier_abs = find_outlier_org(name,0)
    _, outlier_org = find_outlier_org(name,1)
    _, outlier_neigh = find_bad_nrigh(name,outlier_abs)
    
    
    _, outlier_over = find_outlier(name,2)
    #_, outlier_org_over = find_outlier_org(name,2)
    
    _, outlier_neigh_over = find_bad_nrigh(name,outlier_over)

    
    bb = 1*(outlier_neigh<2)*(outlier_neigh_over<20)
    
    return fiber_sub, bb

def gen_manual_label(name_list):
    manual_label_dict = {}
    for i in range(len(name_list)):
        name = name_list[i]
        _, final = final_outlier_modi(name)
        manual_label_dict.update({os.path.splitext(name)[0]: final})
        print('the ' +str(i)+'/144 finished')
        
    np.save('manual_label_dict_modi.npy', manual_label_dict) 
    return manual_label_dict        
        

gen_manual_label(name_list)










#%%
manual_label_dict = np.load('manual_label_dict.npy',allow_pickle = True).item()



#fiber_sub, final = final_outlier(name_list[0])
#%%

#visualize_streamline(fiber_sub,final)

#%%
#visualize_streamline_removed(fiber_sub,final,removed=1)

#visualize_streamline(fiber_sub,final_neigh)
#%%

def find_overlap(a,b):
    print( sum(a*b*2)/(sum(a)+sum(b)))

def find_overlap_3(a,b,c):
    print( sum(a*b*c*3)/(sum(a)+sum(b)+sum(c)))





