# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 12:42:31 2020

@author: CeLiang

fiber clear using tract mask

"""
import  numpy as np
import os
from LUT_labels import  find_index
import edt

path = 'D:\\OneDrive - Law Firm1\\thesis\\Lab\\data\\label'
#filenames = ['601127.npy', '599469.npy','599671.npy' , '613538.npy','620434.npy', '622236.npy']

#name = os.path.join(path, filenames[0])
#%%

#test_mask = np.load(name)

label_index = []
label_index.append(find_index('CC_3'))
label_index.append(find_index('CC_4'))
label_index.append(find_index('CC_5'))
#%%

def gen_mask_tract(index_list,subject_name):
    result = np.zeros((145,174,145))
    tmp = np.load(subject_name)
    for index in index_list:
        result = result + tmp[:,:,:,index]
    result = 1*(result==0)
    dist_mat = edt.edt(result ,anisotropy=(1,1.2,1), black_border=False, order='K', parallel=1)
    return dist_mat




#%%
namelist = np.load('subject_list.npy')

def gen_mask_from_all(namelist):
    result = np.zeros((145,174,145))
    for name in namelist:
        name = str(name)+'.npy'
        name_load = os.path.join(path, name)
        tmp = gen_mask_tract(label_index ,name_load)
        result = result + tmp
    return result
        
mask_tract = gen_mask_from_all(namelist)
#%%
avg_dist = mask_tract / 105 


#%%
np.save('avg_dist',avg_dist)
'''
#%%
test_region = test_mask[:,:,:,label_index[0]]
test_slice = test_region[100,:,:]
test = 1- test_slice
test = test.astype(np.uint32)
#%%
test_dt = edt.edt(test,anisotropy=(1.2,1), black_border=False, order='K', parallel=1) 


#%%
test_3d = 1 - test_region
dt_3d = edt.edt(test_3d ,anisotropy=(1,1.2,1), black_border=False, order='K', parallel=1)
slice_check = dt_3d[100,:,:]

#%%
aa = test_dt - slice_check
 
def gen_mask_tract(index_list,subject_name):
    result = np.zeros((145,174,145))
    tmp = np.load(subject_name)
    for index in index_list:
        result = result + tmp[:,:,:,index]
    result = 1*(result==0)
    dist_mat = edt.edt(result ,anisotropy=(1,1.2,1), black_border=False, order='K', parallel=1)
    return result,dist_mat


namelist = np.load('subject_list.npy')
names = namelist[0]
result = np.zeros((145,174,145))
for name in namelist:
    name = str(name)+'.npy'
    name_load = os.path.join(path, name)
    mask_mat,dist_mat  = gen_mask_tract(label_index ,name_load)

#%%
import matplotlib as mpl
import matplotlib.pyplot as plt

index = 90
mask_mat_s = mask_mat[:,index,:]
dist_mat_s = dist_mat[:,index,:]

fig, ax = plt.subplots()
im = ax.imshow(mask_mat_s,cmap='gray')
ax.axis('off')
plt.colorbar(im)
plt.show()

fig, ax = plt.subplots()
im = ax.imshow(dist_mat_s)
ax.axis('off')
plt.colorbar(im)
plt.show()

'''























