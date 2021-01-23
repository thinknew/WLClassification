import numpy as np
import scipy.io as sio
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger,EarlyStopping
from modelRun import *
from op import getInputDataInfo

import tensorflow.compat.v1 as tf

start=7 # Start from index 0
totalDataLength=8 # Equal to number of datasets
numOfEpochs=300 #300
scaleFactor = 1000  # Fix parameter
numOfKernels = 1  # Fix parameter
visibleGPU="1"
patience=300
delta=0
dropoutRate=0.5
randomState=0
CV_index=4
folder='CV/e'



# # DeepNet
for i in range(start, totalDataLength,1 ):

    LoadMatFileName, Path, dataVar, labelVar, SaveMatFileName, BS, samplingRate, numOfClasses = getInputDataInfo(i)

    kernelLength = (int)(samplingRate / 2)  # Fix parameters which is half of data sampling rate
    modelRun(Path, LoadMatFileName,dataVar,labelVar,numOfClasses,numOfKernels, scaleFactor,
                 BS, checkpointCallbacks(folder+'/DeepConvNet'+SaveMatFileName, patience,delta), folder+'/DeepConvNet'+SaveMatFileName, numOfEpochs,
             samplingRate, "DeepConvNet",dropoutRate,visibleGPU,randomState,CV_index)
