# Anomalous-Curves-Detection

Requirment:
1. tensorflow 2.2.0
2. dipy
3. nibabel
4. fury 
5. plyfile

## Quick detection

python Detection.py -f 156031_ex_cc-body_shore.ply 

```
model_type          = 'deep Bi GRU',default choice 
file_name           = '156031_ex_cc-body_shore.ply'
threshold control   = 80, default value 
length  control     = 40, default value
```


-  -m    Specify the model
-  -f    Input the file name
-  -t    Specify the threshold percentile
-  -l    Specify length under what value is anomalous

