# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 12:30:58 2020

@author: CeLiang
"""
import argparse
from plyfile import PlyData
import numpy as np
import os
import tensorflow as tf
from dipy.tracking.streamline import Streamlines
from dipy.tracking.streamlinespeed import length
from sklearn.metrics import mean_squared_error


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
            


def npz2ply_cleaned(data,name):
       
    data = data.astype(np.float32)

    ply_idx = []
    fiber = data[0]
    fiber = zero_remove(fiber)
    ply_idx.append(np.shape(fiber)[0])
    
    for i in range(1, np.shape(data)[0]):
        tmp = data[i]
        tmp = zero_remove(tmp)
        ply_idx.append(np.shape(tmp)[0]+ply_idx[-1])
        fiber = np.concatenate((fiber,tmp))
        
    
    write_ply_data(os.path.join('result',os.path.splitext(name)[0]+'_cleaned_m'+str(model_type)+'.ply'), fiber, ply_idx, ['x','y','z'])


def ply2np(name):
    # convert a ply format to the matrix we are using
    # input : '128127_ex_cc-body_shore.ply', name of a oly file
    # optput: a matrix (#_of_fibers, #_of_vertex, 3 )
    temo = PlyStruct()
    temo.load_poly_data(os.path.join('data',name),num_prop=3)
    #data = temo.ply_data
  
    temo.gen_stream_line()
    stream = temo.stream_line


    stream = np.array(stream)
    len_table = get_length(stream)
    subject = np.zeros((len(stream),np.max(len_table),3))
    for i in range(len(stream)):
        length = len_table[i]
        subject[i,0:length,:] = stream[i]
        
    return subject




model_list = ['deep_bi_GRU.h5', 'deep_bi_LSTM.h5', 'deep_GRU.h5', 'deep_LSTM.h5', 
              'GRU_bi_model.h5', 'GRU_dr_model.h5', 'GRU_model.h5', 'LSTM_bi_model.h5', 
              'LSTM_dr_model.h5', 'LSTM_model.h5', 'simple_rnn_2_model.h5', 
              'sRNN_bi_model.h5', 'toy_rnn_2_dr_model.h5']

titles = ['deep Bi GRU', 'deep Bi LSTM', 'deep GRU', 'deep LSTM', 
          'Bi GRU.h5', 'Undercomplete GRU', 'Autoencoder GRU', 'Bidirectional LSTM', 
          'Undercomplete LSTM', 'Autoencoder LSTM', 'Autoencoder RNN', 
          'Bidirectional RNN', 'Undercomplete RNN']

#%%
parser = argparse.ArgumentParser()

parser.add_argument('-m', action='store',
                    dest='model_type',
                    type=int,
                    default = 0,
                    help='Specify the model')

parser.add_argument('-f', action='store',
                    dest='file_name',
                    help='Input the file name')

parser.add_argument('-t', action='store',
                    dest='thre_con',
                    default = 80,
                    help='Specify the threshold percentile')

parser.add_argument('-l', action='store',
                    dest='len_con',
                    default = 40,
                    help='Specify length under what value is anomalous')

parser.add_argument('--version', action='version',
                    version='%(prog)s 1.0')

results = parser.parse_args()
print('model_type          = {!r}'.format(titles[results.model_type]))
print('file_name           = {!r}'.format(results.file_name))
print('threshold control   = {!r}'.format(results.thre_con))
print('length  control   = {!r}'.format(results.len_con))

name = results.file_name
bundle = ply2np(name)
thre_con = results.thre_con
model_type = results.model_type
#%%


model_path = os.path.join('trained_models',model_list[results.model_type])

model = tf.keras.models.load_model(model_path,custom_objects={'leaky_relu': tf.nn.leaky_relu})


denoise = model(bundle)
denoise = denoise.numpy()
mse_idv = cal_mse(bundle, denoise)



thre = np.percentile(mse_idv,thre_con)
pred_1 = 1*(mse_idv<thre)
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



op_path = 'result'
if not os.path.exists(op_path):
    os.mkdir(op_path)

save_path = os.path.join(op_path,'Detection'+name.split('_')[0]+'_m'+str(results.model_type))
np.save(save_path,pred)

data_new = np.delete(bundle, np.where(pred == 0), axis=0)

npz2ply_cleaned(data_new,name)


