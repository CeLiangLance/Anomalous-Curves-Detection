# Anomalous-Curves-Detection

Requirment:
1. tensorflow 2.2.0
2. dipy
3. nibabel
4. fury 
5. plyfile

Running tip: create a **tensorflow-cpu** environment to run is safer since you might have a memory problem if you are using tensorflow-gpu and your gpu memory is not high. 

## Quick detection

```python Detection.py -f 156031_ex_cc-body_shore.ply ```

>the terminal will print the following and generate the results\
model_type          =  0,default choice\
file_name           = '156031_ex_cc-body_shore.ply'\
threshold control   = 80, default value\
length  control     = 40, default value

If you run this sample, you could find two files in the folder /result\
-156031_ex_cc-body_shore_cleaned_m0.ply\
the anomaly removed version  
-Detection156031_m0.npy\
a list of the detection result 1：normal 0：anomaly 





-  -m    Specify the model\
Default: 0. the model choice will be explained later
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
You can choose one of the following models\
models 0，1，2，3，6，9，10 has better result.\
The implementation of these models could be found in /models. 
```
index: 0       model: deep Bi GRU
index: 1       model: deep Bi LSTM
index: 2       model: deep GRU
index: 3       model: deep LSTM
index: 4       model: Bi GRU
index: 5       model: Undercomplete GRU
index: 6       model: Autoencoder GRU
index: 7       model: Bidirectional LSTM
index: 8       model: Undercomplete LSTM
index: 9       model: Autoencoder LSTM
index: 10      model: Autoencoder RNN
index: 11      model: Bidirectional RNN
index: 12      model: Undercomplete RNN
```

## The Seq Related Models 
The Seq2seq related models are implemented in a different way, so we create a new folder /Seq_models to store the weights and code.
the **-Detection.ipynb demonstrated how to generate the detection result.

## Anomaly Detection Performance
In performance_analysis.ipynb, we demonstrated how to evaluate the Anomaly detection performance between different models and with the manually Detection results. 
