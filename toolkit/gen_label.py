# -*- coding: utf-8 -*-
"""
Created on Sun Nov  1 08:52:38 2020

@author: CeLiang
"""


import  numpy as np
import os
from visualize_score import  visualize_streamline ,visualize_streamline_removed

import matplotlib as mpl
import matplotlib.pyplot as plt

#%%
name_list = np.load(os.path.join('..','data_4_model_new', 'filenames.npy'),'r')
test_list = np.load(os.path.join('..','data_4_model_new', 'valid_list.npy'),'r')

mask_score_dict = np.load('mask_score_dict.npy',allow_pickle=True).item()
#IFscore_dict = np.load('IFscore_dict_17.npy',allow_pickle=True).item()
cci_dict = np.load('cci_dict.npy',allow_pickle=True).item()
ALLscore_dict = np.load('ALLscore_dict_10.npy',allow_pickle=True).item()





#%%


def find_outlier(name):
    # thre: 0 absolute outlier
    #       1 high prob outlier
    key = os.path.splitext(name)[0]
    mask_score = mask_score_dict[key]
    mask_score = np.array(mask_score)

    cci_score = cci_dict[key]
    cci_score = np.array(cci_score)

    ALL_score = ALLscore_dict[key]
    ALL_score = np.array(ALL_score)
    #ALL_score = ALL_score +1
    
    fiber_sub = np.load(os.path.join('..','cci_clean_data',name))
    fiber_sub = fiber_sub['arr_0']
    
    mask_score_test = 1*(mask_score<630)

    cci_score_test = 1* (cci_score>2)
    
    ALL_score_test = 1*(ALL_score>0)
    
    final = cci_score_test * mask_score_test * ALL_score_test

    
    return fiber_sub, final


a, b = find_outlier(name_list[0])



#%%

def gen_manual_label(name_list):
    manual_label_dict = {}
    for i in range(len(name_list)):
        name = name_list[i]
        _, final = find_outlier(name)
        manual_label_dict.update({os.path.splitext(name)[0]: final})
        print('the ' +str(i)+'/144 finished')
        
    np.save('manual_label_dict_84.npy', manual_label_dict) 
    return manual_label_dict        
        

gen_manual_label(name_list)


    
#%%






name = test_list[0]

cci = cci_dict[os.path.splitext(name)[0]]

test_data = np.load(os.path.join('..','cci_clean_data', name),'r')
data_test = test_data['arr_0']
#%%


cci = cci_dict[os.path.splitext(name_list[0])[0]]
for i in range(1,len(name_list)):
    tmp = cci_dict[os.path.splitext(name_list[i])[0]]
    cci = np.concatenate((cci,tmp))

mask = mask_score_dict[os.path.splitext(name_list[0])[0]]
for i in range(1,len(name_list)):
    tmp = mask_score_dict[os.path.splitext(name_list[i])[0]]
    mask = np.concatenate((mask,tmp))
    

    
allscore = ALLscore_dict[os.path.splitext(name_list[0])[0]]
for i in range(1,len(name_list)):
    tmp = ALLscore_dict[os.path.splitext(name_list[i])[0]]
    allscore = np.concatenate((allscore,tmp))
    
#%%
name = test_list[0]
cci_0 = cci_dict[os.path.splitext(name)[0]]
mask_0 = mask_score_dict[os.path.splitext(name)[0]]
allscore_0 = ALLscore_dict[os.path.splitext(name)[0]]


test_data = np.load(os.path.join('..','cci_clean_data', name),'r')
data_test = test_data['arr_0']

cci_0_thr = 1*(np.array(cci_0)>2)
mask_0_thr = 1*(np.array(mask_0)<630)
allscore_0_thr = 1*(np.array(allscore_0)>0)

#%%
mask_0 = np.array(mask_0)
mask_0_reverse = 800 - mask_0
for i in range(len(mask_0_reverse)):
    if mask_0_reverse[i]<0 : mask_0_reverse[i]=0

#
visualize_streamline(data_test,mask_0_reverse,control_par=4.23)



#%%
visualize_streamline(data_test,np.array(cci_0_thr),control_par=1)
#%%
visualize_streamline_removed(data_test,cci_0_thr)

#%%
font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 17,
         }

fig, ax = plt.subplots(1,figsize=(6.5, 4.5))
ax.hist(cci, bins=1000,histtype='stepfilled')
ax.axis([0, 100, 0, 4600])
ax.axvline(x=2, color="r", linestyle="--")
ax.tick_params(labelsize=23)

ax.set_xlabel('CCI', font2)
ax.set_ylabel('# streamlines', font2)  


#%%
fig, ax = plt.subplots(1,figsize=(6.5, 4.5))
ax.hist(mask, bins=300, histtype='stepfilled')
ax.axis([0, 1000, 0, 20000])
ax.axvline(x=630, color="r", linestyle="--")
ax.tick_params(labelsize=13)

ax.set_xlabel('Distance Score', font2)
ax.set_ylabel('# streamlines', font2)  

#%%
cci_thr = 1*(cci>2)
mask_thr = 1*(mask<630)

allscore = 1*(allscore>0)

print(sum(mask_thr*allscore) / sum(allscore))
print(sum(mask_thr*allscore) / sum(mask_thr))

print(sum(mask_thr*allscore) / sum(1*(mask_thr+allscore)>0))

print(sum(cci_thr*mask_thr*allscore) / len(cci_thr))
print(sum(1*(cci_thr+mask_thr+allscore)>2) / len(cci_thr))

#%%
name =  test_list[0]
cci_0 = cci_dict[os.path.splitext(name)[0]]

fig, ax = plt.subplots(1)
ax.hist(cci_0, bins=50, histtype='step')
ax.set_xlabel('CCI')
ax.set_ylabel('# streamlines')  

#%%
cci_10 = cci_dict[os.path.splitext(test_list[10])[0]]
cci_20 = cci_dict[os.path.splitext(test_list[20])[0]]



font2 = {'family': 'Times New Roman',
         'weight': 'normal',
         'size': 20,
         }

fig, ax1 = plt.subplots(figsize=(8, 6))

ax1.set_xlabel("CCI",font2,labelpad = 1)
ax1.set_ylabel('# streamlines', font2,labelpad = 1)  

ax1.hist(cci_0, bins=50, histtype='step',label='subject ID: '+ os.path.splitext(test_list[0])[0])

ax1.hist(cci_10, bins=50, histtype='step', label='subject ID: '+os.path.splitext(test_list[10])[0])

ax1.axis([0, 65, 0, 125])
ax.tick_params(labelsize=23)
ax1.legend(fontsize=16,  ##设置图例文字大小
          title_fontsize=14)  ##设置图例title大小)
#ax1.hist(cci_20, bins=50, histtype='step')

plt.show()
'''


ifscore = IFscore_dict[os.path.splitext(name_list[0])[0]]
for i in range(1,len(name_list)):
    tmp = IFscore_dict[os.path.splitext(name_list[i])[0]]
    ifscore = np.concatenate((ifscore,tmp))

'''





