# Anomalous-Curves-Detection

Requirment:
1. tensorflow 2.2.0
2. dipy
3. nibabel
4. fury 
5. plyfile

## Quick detection

```python Detection.py -f 156031_ex_cc-body_shore.ply ```

>the terminal will print the following and generate the result in /result\
model_type          = 'deep Bi GRU',default choice\
file_name           = '156031_ex_cc-body_shore.ply'\
threshold control   = 80, default value\
length  control     = 40, default value



-  -m    Specify the model\
the model choice will be explained later
-  -f    Input the file name\
the file is in /data folder, there is an sample   
-  -t    Specify the threshold percentile\
Default: 85. The Autoencoder reconstruct the input fibers and we calculate the reconstruction MSE, rank the MSE and set a percentile as the detection threshold.  
-  -l    Specify length under what value is anomalous\
Default: 40. 

## Visualization 
Detection_demo.ipynb offers two methods to visualize the result:
1. a binary visualization of the detection\
red: normal curves\
white : anomalous curves
2. anomalous fibers removal

![test](/pic/tt.png "two methods")
## Model Choice 
```
index: 0       model: deep Bi GRU
index: 1       model: deep Bi LSTM
index: 2       model: deep GRU
index: 3       model: deep LSTM
index: 4       model: Bi GRU.h5
index: 5       model: Undercomplete GRU
index: 6       model: Autoencoder GRU
index: 7       model: Bidirectional LSTM
index: 8       model: Undercomplete LSTM
index: 9       model: Autoencoder LSTM
index: 10      model: Autoencoder RNN
index: 11      model: Bidirectional RNN
index: 12      model: Undercomplete RNN
```
