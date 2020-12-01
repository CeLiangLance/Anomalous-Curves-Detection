
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
import argparse
from plyfile import PlyData

#%%
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
    # to get the length of streamlines in this subject
    len_table = []
    for i in range(len(stream)):
        len_table.append(np.shape(stream[i])[0])
    return len_table




def ply2np(name):
    # convert a ply format to the matrix we are using
    # input : '128127_ex_cc-body_shore.ply', name of a oly file
    # optput: a matrix (#_of_fibers, #_of_vertex, 3 )
    temo = PlyStruct()
    temo.load_poly_data(os.path.join(name),num_prop=3)
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


parser = argparse.ArgumentParser()



parser.add_argument('-f', action='store',
                    dest='file_name',
                    help='Input the file name')



results = parser.parse_args()

print('file_name           = {!r}'.format(results.file_name))



# 'Raw_156031_ex_cc-body_shore.ply'
name = results.file_name 
bundle = ply2np(name)
#%%
data, _ = clean_subject(bundle)
name_save = os.path.join('..','cci_clean_data',os.path.splitext(name)[0])
#np.savez_compressed(name_save,data)





def write_ply_data(output_path, ply_data, ply_idx, property_names):
    """
    Write ply data to ply file.
    :param str output_path: output file path
    :param ndarray ply_data: 2D array with vertex data
    :param ndarray ply_idx: 1D array with end indices of fibers
    :param list property_names: list of property names, len should be equal to ply_data.shape[1]
    """
    property_string = ''.join([f'property float {i}\n' for i in property_names])
    header = f"comment DTI Tractography, produced by mrtrix tckgen\n" \
        f"element vertices {len(ply_data)}\n" \
        f"{property_string}" \
        f"element fiber {len(ply_idx)}\n" \
        f"property int endindex\n" \
        f"end_header\n"

    with open(output_path, 'w') as file:
        file.write("ply\nformat ascii 1.0\n")
        file.write(header)
        for ply_vertex in ply_data:
            file.write("{} {} {}\n".format(ply_vertex[0],ply_vertex[1],ply_vertex[2]))
        for stream_id in ply_idx:
            file.write(f"{stream_id}\n")
            
def npz2ply_org(data,name):
    

    ply_idx = []
    fiber = data[0]
    fiber = zero_remove(fiber)
    ply_idx.append(np.shape(fiber)[0])
    
    for i in range(1, np.shape(data)[0]):
        tmp = data[i]
        tmp = zero_remove(tmp)
        ply_idx.append(np.shape(tmp)[0]+ply_idx[-1])
        fiber = np.concatenate((fiber,tmp))
        
    
    write_ply_data(name, fiber, ply_idx, ['x','y','z'])



#%%

name_save = '_'.join(name.split("_")[1:])

npz2ply_org(data,name_save)





