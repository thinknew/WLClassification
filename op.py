from tensorflow.keras import utils as np_utils
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger,EarlyStopping
from sklearn.metrics import mean_squared_error,f1_score, accuracy_score,average_precision_score,\
    precision_score,recall_score,mean_absolute_error,cohen_kappa_score


def getClassInfo(numOfClasses):
    # Selecting the class
    if numOfClasses == 2:
        f1_avg = 'binary'
        pos_label = 0
        loss_type = 'binary_crossentropy'
        class_weights = {0: 1, 1: 1}  # For two classes
    elif numOfClasses == 3:
        f1_avg = 'weighted'
        pos_label = 1
        loss_type = 'categorical_crossentropy'
        class_weights = {0: 1, 1: 1, 2: 2}  # For three classes
    elif numOfClasses == 4:
        f1_avg = 'weighted'
        pos_label = 1
        loss_type = 'categorical_crossentropy'
        class_weights = {0: 1, 1: 1, 2: 2, 3: 1}  # For four classes
    else:
        f1_avg = 'weighted'
        pos_label = 1
        loss_type = 'categorical_crossentropy'
        class_weights = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1}  # For five classes

    return f1_avg,pos_label,loss_type,class_weights



def getInputDataInfo(index):

    # CC , pHRC, WL, MI, P300, MCR, ECG, HRV
    # Add data related information as required
    LoadMatFileName = ['cleanedPredictionData.mat','cleanedTrackingData.mat','rawPredictionData.mat',
                       'FOG_VoluntaryStop.mat',
                       'S6.mat','S7.mat','dataWorkload_Without_Walking.mat', 'binary_data.mat','MINDO_SSVEP.mat','dataForML_three_classes.mat']

    Path = ['/data/avisingh/Desktop/Alka_Data/ATC/','/data/avisingh/Desktop/Alka_Data/ATC/',
            '/data/avisingh/Desktop/Alka_Data/ATC/','/data/avisingh/Desktop/Alka_Data/FOG/',
            '/data/avisingh/Desktop/CAS/AllData/','/data/avisingh/Desktop/CAS/AllData/','/data/avisingh/Desktop/Workload_Thongs/Six_Class_data/',
            '/data/avisingh/Desktop/Workload_Thongs/Set_Data/Epoch_Cleaned/','/data/avisingh/Desktop/Sai_SSVEP/','/data/avisingh/Desktop/CAS/']

    dataVar = ['predictionData','trackData','predictionData','data',
        'Data','Data','data', 'binary_data','data', 'CC']
    labelVar = ['predictionLabels','trackLabels','predictionLabels','labels',
        'Lables','Lables','labels', 'Label_two_class','labels','Labels']
    SaveMatFileName = ['CPD','CTD','RPD','FOG',
        'S6','S7','Thong_Without_Walking', 'Thong_Binary_Class','Sai_SSVEP', 'CAS_3C']

    samplingRate = [250,250,250,500,
                    1000,1000,250, 250, 500, 1000]
    BS = [16,16,16,10,
          16,16,16,16,6,16]
    numOfClasses = [3,3,3,2,
                    2,2,4, 2, 3,3]

    return LoadMatFileName[index],Path[index],dataVar[index],\
           labelVar[index],SaveMatFileName[index],BS[index],samplingRate[index],numOfClasses[index]




def getPerformanceMetricsDL(numOfClasses, pos_label,f1_avg ,Y_test, predicted):

    mse = mean_squared_error(Y_test.argmax(axis=-1), predicted.argmax(axis=-1))
    mae = mean_absolute_error(Y_test.argmax(axis=-1), predicted.argmax(axis=-1))
    co_kap_sco = cohen_kappa_score(Y_test.argmax(axis=-1), predicted.argmax(axis=-1))
    acc = accuracy_score(Y_test, predicted)

    if numOfClasses == 2:

        predicted = predicted.argmax(axis=-1)
        Y_test = Y_test.argmax(axis=-1)

        avg_pre_sco = average_precision_score(Y_test, predicted, average='weighted', pos_label=0)
        precision = precision_score(Y_test, predicted, average=f1_avg, pos_label=pos_label)
        recall = recall_score(Y_test, predicted, average=f1_avg, pos_label=pos_label)
        f1_sc = f1_score(Y_test, predicted, average=f1_avg, pos_label=pos_label)
    else:
        avg_pre_sco = average_precision_score(Y_test, predicted, average='weighted')
        precision = precision_score(Y_test, predicted, average=f1_avg)
        recall = recall_score(Y_test, predicted, average=f1_avg)
        f1_sc = f1_score(Y_test, predicted, average=f1_avg)

    return mse,mae,co_kap_sco,acc,avg_pre_sco,precision,recall,f1_sc

def oneHot(input, numOfClasses,ravelBool):

    if ravelBool:
        output=np_utils.to_categorical(input.ravel(), numOfClasses)
    else:
        output=np_utils.to_categorical(input, numOfClasses)

    return output

# Divide data into 50/25/25 for train, val, and test
def getTrainTestVal(X,Y,testSize=0.3,randomState=0,CV_index=0):

    train_test_data = StratifiedShuffleSplit(n_splits=5, test_size=testSize, random_state=randomState)

    # Dividing into train and test
    lists=list(train_test_data.split(X, Y))
    train,test=lists[CV_index]
    # for train, test in train_test_data.split(X, Y):

    X_train = X[train]
    Y_train = Y[train]

    # Dividing into train and validation

    X_test = X[test]
    Y_test = Y[test]


    return X_train,X_test,Y_train,Y_test

def checkpointCallbacks(SaveMatFileName, patience, min_delta):

    checkpoint = [
        EarlyStopping(monitor='val_loss', patience=patience, min_delta=min_delta, mode="auto", restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=SaveMatFileName+'.ckpt', monitor='loss', mode="auto", save_weights_only=True, verbose=1),
        CSVLogger(SaveMatFileName + '.csv', append=True),
        ]
    return checkpoint

