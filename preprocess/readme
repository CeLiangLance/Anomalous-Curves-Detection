
# Manually_Detection
python Manually_Detection.py -h
usage: Manually_Detection.py [-h] [-f FILE_NAME] [-IF IF_THRE] [-dis DIS_THRE] [-CCI CCI_THRE] [-l LEN_CON]

  -h, --help     show this help message and exit
  -f FILE_NAME   Input the file name
  -IF IF_THRE    Specify the Isolation Forest threshold, default 0.17
  -dis DIS_THRE  Specify the distance score threshold, defalut 630
  -CCI CCI_THRE  Specify the CCI threshold default 2
  -l LEN_CON     Specify length under what value is anomalous

IF_THRE: see contamination from <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html>/
DIS_THRE and CCI_THRE are set by distribution of distance score and CCI.
the output file is saved here.

### the file for Manually Detection is in Anomalous-Curves-Detection/data/, not here!
the output file is DetectionXXXX_manual.npy, XXXX is the subject ID.



#Before Manually_Detection
to run the Manually_Detection.py, the ply should be cleaned the indentical fibers, or the CCI calculation will have poroblem
python indentical_clean.py -f  'Raw_156031_ex_cc-body_shore.ply'

### the input and putput file are #both# in the floder Anomalous-Curves-Detection/preprocess , Be carful!
