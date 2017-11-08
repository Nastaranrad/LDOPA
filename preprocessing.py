
"""
@author: nastaran.mrad@gmail.com 

########## this script prepares data for one-subject-leave-out scenario by:
            1) download data
            2) remove missing values
            3) segment data into 1-sec (50 time points)
            4) construct raw data appropriate for one-subject-leave-out scenario
#########
"""
from pandas import DataFrame
import synapseclient
import pandas as pd
import numpy as np
import scipy.io as sio
import pickle
import csv
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from copy import deepcopy
from itertools import repeat
import os
from sklearn.preprocessing import Imputer


experiment = 'Bradykinesia' # Dyskinesia/Bradykinesia/Tremor

############## download  accelerometer data ###
syn = synapseclient.login()
query_ldopa_tabel = syn.tableQuery("SELECT 'dataFileHandleId' FROM syn10495809")
ldopa_dataFileHandleId_DataFrame = query_ldopa_tabel.asDataFrame()
prev_cache_loc = syn.cache.cache_root_dir
syn.cache.cache_root_dir = 'your preference path/' 
tsv_files = syn.downloadTableColumns(query_ldopa_tabel,"dataFileHandleId")
syn.cache.cache_root_dir = prev_cache_loc


############################

items = tsv_files.items()
activities_files_temp = pd.DataFrame({"dataFileHandleId": [i[0] for i in items], "dataFileHandleId_tsv_filePath": [i[1] for i in items]})

#################### read whole LDOPA tabel ###################################
query_ldopa_tabel = syn.tableQuery("SELECT * FROM syn10495809")
ldopa_DataFrame = query_ldopa_tabel.asDataFrame()
# convert ints to strings for merging
ldopa_DataFrame["dataFileHandleId"] = ldopa_DataFrame["dataFileHandleId"].astype(str)
ldopa_DataFrame["session"] = ldopa_DataFrame["session"].astype(str)
ldopa_DataFrame["visit"] = ldopa_DataFrame["visit"].astype(str)
ldopa_DataFrame["tremorScore"] = ldopa_DataFrame["tremorScore"].astype(str)
ldopa_DataFrame["dyskinesiaScore"] = ldopa_DataFrame["dyskinesiaScore"].astype(str)
ldopa_DataFrame["bradykinesiaScore"] = ldopa_DataFrame["bradykinesiaScore"].astype(str)
actv_temp = pd.merge(ldopa_DataFrame, activities_files_temp, on="dataFileHandleId")

pickle.dump(actv_temp, open("your save path/" + "_OriginalData_DataFrame.p" , "wb"))

##################preprocessing################################################
FSAMP = 50
Low = 0.1
High = 20 
def butter_bandpass_filter(data, lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

################ change the path to the preproceesed data path ################
preprocessed_data = deepcopy(actv_temp)
for i in range(actv_temp.shape[0]):
    path = actv_temp["dataFileHandleId_tsv_filePath"][i]
    preprocessed_data["dataFileHandleId_tsv_filePath"][i]= path.replace("/Data/", "/Preprocessed_Data/", 1)
    
########## impute missing values and remove files with missing labels ################
if experiment == 'Bradykinesia':
    score = 'bradykinesiaScore'
    i = 0
    j = 0    
    for row in preprocessed_data["dataFileHandleId_tsv_filePath"]:
        with open(row) as tsvin:
            df = pd.read_table(tsvin, skipinitialspace=True)        
            df = df.dropna(axis = 0, how = 'any')
            if df.empty:                    
                preprocessed_data = preprocessed_data.drop(preprocessed_data.index[i-j])
                j += 1
            i += 1
            print ('i %d:' %(i)) 
    preprocessed_data = preprocessed_data.loc[preprocessed_data[score] != 'nan']
    preprocessed_data.index = pd.RangeIndex(len(preprocessed_data.index))      
    i = 0
    for row in preprocessed_data["dataFileHandleId_tsv_filePath"]:
        with open(row) as tsvin:
            imputer=Imputer(missing_values='NaN', strategy='mean', axis=0)
            df = pd.read_table(tsvin, skipinitialspace=True)        
            if df.isnull().values.any():
                imputer = imputer.fit(df)
                df=imputer.transform(df)
                df = pd.DataFrame(df)
                df.to_csv(preprocessed_data["dataFileHandleId_tsv_filePath"][i],sep='\t', index = False)
                print(preprocessed_data["dataFileHandleId_tsv_filePath"][i])
            i += 1
            print ('i %d:' %(i))
    pickle.dump(preprocessed_data, open(" your save path "+  experiment + "_DataFrame_Removed_NanFiles_NanLabels.p" , "wb"))

    
elif experiment == 'Dyskinesia':
    score = 'dyskinesiaScore'
    i = 0
    j = 0    
    for row in preprocessed_data["dataFileHandleId_tsv_filePath"]:
        with open(row) as tsvin:
            df = pd.read_table(tsvin, skipinitialspace=True)        
            df = df.dropna(axis = 0, how = 'any')
            if df.empty:                    
                preprocessed_data = preprocessed_data.drop(preprocessed_data.index[i-j])
                j += 1
            i += 1
            print ('i %d:' %(i)) 
    preprocessed_data = preprocessed_data.loc[preprocessed_data['task'].isin(['ramr1','ramr2','raml1','raml2','ftnl1','ftnl2','ftnr1','ftnr2'])]
    preprocessed_data = preprocessed_data.loc[preprocessed_data[score] != 'nan']
    preprocessed_data.index = pd.RangeIndex(len(preprocessed_data.index))      
    i = 0
    for row in preprocessed_data["dataFileHandleId_tsv_filePath"]:
        with open(row) as tsvin:
            imputer=Imputer(missing_values='NaN', strategy='mean', axis=0)
            df = pd.read_table(tsvin, skipinitialspace=True)        
            if df.isnull().values.any():
                imputer = imputer.fit(df)
                df=imputer.transform(df)
                df = pd.DataFrame(df)
                df.to_csv(preprocessed_data["dataFileHandleId_tsv_filePath"][i],sep='\t', index = False)
                print(preprocessed_data["dataFileHandleId_tsv_filePath"][i])
            i += 1
            print ('i %d:' %(i))
    pickle.dump(preprocessed_data, open("your save Path/"+  experiment + "_DataFrame_Removed_NanFiles_NanLabels.p" , "wb"))

elif experiment == 'Tremor':
    score = 'tremorScore'
    i = 0
    j = 0    
    for row in preprocessed_data["dataFileHandleId_tsv_filePath"]:
        with open(row) as tsvin:
            df = pd.read_table(tsvin, skipinitialspace=True)        
            df = df.dropna(axis = 0, how = 'any')
            if df.empty:                    
                preprocessed_data = preprocessed_data.drop(preprocessed_data.index[i-j])
                j += 1
            i += 1
            print ('i %d:' %(i)) 
    preprocessed_data = preprocessed_data.loc[preprocessed_data[score] != 'nan']
    preprocessed_data.index = pd.RangeIndex(len(preprocessed_data.index)) 
    i = 0
    for row in preprocessed_data["dataFileHandleId_tsv_filePath"]:
        with open(row) as tsvin:
            imputer=Imputer(missing_values='NaN', strategy='mean', axis=0)
            df = pd.read_table(tsvin, skipinitialspace=True)        
            if df.isnull().values.any():
                imputer = imputer.fit(df)
                df=imputer.transform(df)
                df = pd.DataFrame(df)
                df.to_csv(preprocessed_data["dataFileHandleId_tsv_filePath"][i],sep='\t', index = False)
                print(preprocessed_data["dataFileHandleId_tsv_filePath"][i])
            i += 1
            print ('i %d:' %(i))
    pickle.dump(preprocessed_data, open("your save path/"+  experiment + "_DataFrame_Removed_NanFiles_NanLabels.p" , "wb"))


##### make a seperated dataframe for each subject ################ 
subjects = []
patients = preprocessed_data.patient.unique() 
for i in range(patients.shape[0]):
    temp_sub = preprocessed_data.loc[preprocessed_data["patient"] == patients[i]]
    temp_sub.index = pd.RangeIndex(len(temp_sub.index))    
    subjects.append(temp_sub)

###### 1) task segmentation 2) concatenate all segmented tasks from different sessions and visits for individual subjects #################################
savePath = 'your save path/'+ experiment + '/'
labels = []
acc_data = []  
all_subjects = []
all_labels = []
sample_length = 1
samplingFreq = 50
overlap = 10
window_size = int(sample_length * samplingFreq)
segmented_data = []
segmented_labels = []
for sub in range(patients.shape[0]):
    i = 0
    acc_data = []
    labels = []
    sample_nums = deepcopy(subjects[sub])
    sample_nums = sample_nums.assign(sampleNum = np.zeros([subjects[sub].shape[0]]))

    for row in subjects[sub]["dataFileHandleId_tsv_filePath"]:
        with open(row) as tsvin:
            df = pd.read_table(tsvin, skipinitialspace=True) 
            temp = (df.as_matrix())[:,1:4]
            sampleNum = (temp.shape[0]/overlap)-samplingFreq+1
            channelNum = temp.shape[1]    
            label = np.zeros([temp.shape[0],1]) 
            label[:,0] = subjects[sub][score][i]
            X = np.zeros([sampleNum,window_size,channelNum])    
            Y = np.zeros([sampleNum,1],dtype=int)            
            for s in range(sampleNum):
                X[s,:,:] = temp[s*overlap:s*overlap+window_size,:]
                Y[s,0] = label[s,]
            sample_nums['sampleNum'][i] = sampleNum # a dataframe with sample_nums column to save the number of samples per each task
            sio.savemat(savePath + experiment + '_subject'+str(sub+1)+ '_task'+ str(i)+'_dataFileHandleId'+subjects[sub]['dataFileHandleId'][i]+'.mat',{'X':X,'label':Y})
            i+= 1
            print("i %d" %(i))            
            
            labels.append(Y)
            acc_data.append(X)            
    sample_nums.to_pickle(savePath + 'subject' + str(sub+1))
    all_subjects.append(np.vstack(acc_data))
    all_labels.append(np.vstack(labels))

###### save each subject seperately##############
subNum = 19
for sub in range(subNum):
    sio.savemat(savePath +'/Validation/'+ experiment + '_subject' + str(sub+1) + '.mat' ,{'X' : all_subjects[sub], 'label': all_labels[sub]})

############# load and concatenate all subjects to create training features ##############
list_subjects = []    
list_labels = []
list_all_subjects = []
list_all_labels = []
for sub in range(subNum):
    matContent = sio.loadmat(savePath +experiment + '_subject' +str(sub+1) + '.mat')
    trainingFeatures = matContent['X']
    trainingLabels = matContent['label']
    list_subjects.append(trainingFeatures)
    list_labels.append(trainingLabels)
del trainingFeatures, trainingLabels
trainingFeatures = np.float32(np.vstack(list_subjects))
trainingLabels = np.int32(np.vstack(list_labels))

sio.savemat(savePath + experiment + '_all_trainingFeatures.mat',{'trainingFeatures':trainingFeatures, 'trainingLabels':trainingLabels})

list_all_subjects.append(np.vstack(list_subjects))
list_all_labels.append(np.vstack(list_labels))    
    
######### one-subject-leave-out#####################################
for sub in range(subNum): 
    # Preparing data
    testFeatures = all_subjects[sub]
    testLabels = all_labels[sub]
    testLabels= testLabels.astype(int)
    testLabels =  np.squeeze(testLabels)
    
    train_index = np.setdiff1d(range(subNum),sub)
    trainingFeatures= all_subjects[train_index[0]]
    trainingLabels = all_labels[train_index[0]]
    train_index = np.setdiff1d(train_index,train_index[0])
    for j in range(len(train_index)):
        trainingFeatures = np.concatenate((trainingFeatures,all_subjects[train_index[j]]),axis = 0)
        trainingLabels = np.concatenate((trainingLabels,all_labels[train_index[j]]),axis = 0)
        
    sio.savemat(savePath+experiment+'_one_subject_out'+str(sub+1)+'.mat',{'trainingFeatures':trainingFeatures,'trainingLabels':trainingLabels,
                 'testFeatures':testFeatures,'testLabels':testLabels})
    del trainingFeatures, trainingLabels, testLabels, testFeatures

