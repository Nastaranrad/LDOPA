
"""
Feature matrix for submitting
@author: nastaran nastaran.mrad@gmail.com
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

data_type = 'Training' # or test
experiment = 'Dyskinesia' # or Bradykinesia/Dyskinesia
path = 'your save path/'+ data_type + '_Segmented_tasks/'+ experiment + '/'

############### read training data for each subjects and concatenate all training data
temp_df = pd.read_pickle(path + 'subject1')
for sub in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]:
    df = pd.read_pickle(path + 'subject' + str(sub+1))
    temp_df = temp_df.append(df,ignore_index = True)
   
############ read and load the best training features and corresponding Labels  ################################
matContent = sio.loadmat('/path of learned features/' + experiment + '/Features/' + experiment +'_Learned_Features_run_9.mat')
trainingFeatures = matContent['trainingFeatures']
trainingLabels = matContent['trainingLabels']
testFeatures = matContent['testFeatures']

############ create a dataframe for the mean/min/max and median of features per each task
d = {"dataFileHandleId":temp_df["dataFileHandleId"]}
submission = pd.DataFrame(d)
i = 0

temp_mean = []
temp_min = []
temp_max = []
temp_median = []

mean_all_tasks = []
min_all_tasks = []
max_all_tasks = []
median_all_tasks = []
for sample in range(temp_df.shape[0]):
    df = trainingFeatures[i:i+np.int(temp_df["sampleNum"][sample]),:]
    ####### mean ###############
    df_mean = np.mean(df,axis = 0)
    df_mean = np.transpose(df_mean[:,np.newaxis])
    ##### min ###############
    df_min = np.min(df,axis = 0 )
    df_min = np.transpose(df_min[:,np.newaxis])
    ##### max #################
    df_max = np.max(df,axis = 0)
    df_max = np.transpose(df_max[:,np.newaxis])
    ###### median #############
    df_median = np.median(df,axis = 0)
    df_median = np.transpose(df_median[:,np.newaxis])
    
    temp_mean.append(df_mean)    
    temp_min.append(df_min)
    temp_max.append(df_max)
    temp_median.append(df_median)
    
    i +=np.int(temp_df["sampleNum"][sample])
    
####### make a list of tasks #####################
mean_all_tasks.append(np.vstack(temp_mean))
min_all_tasks.append(np.vstack(temp_min))
max_all_tasks.append(np.vstack(temp_max))
median_all_tasks.append(np.vstack(temp_median))

########## assign a header for dataframe
columns_mean = ['feature'+str(j+1) for j in range(56)]
columns_median = ['feature'+str(j+1) for j in range(56,112)]
columns_max = ['feature'+str(j+1) for j in range(112,168)]
columns_min = ['feature'+str(j+1) for j in range(168,224)]

########## make a seperate dataframe per mean/max/mean/median of tasks features
df_training_mean = pd.DataFrame(np.array(mean_all_tasks).reshape(np.shape(mean_all_tasks)[1],np.shape(mean_all_tasks)[2]), columns = columns_mean)
df_training_median = pd.DataFrame(np.array(median_all_tasks).reshape(np.shape(median_all_tasks)[1],np.shape(median_all_tasks)[2]), columns = columns_median)
df_training_max = pd.DataFrame(np.array(max_all_tasks).reshape(np.shape(max_all_tasks)[1],np.shape(max_all_tasks)[2]), columns = columns_max)
df_training_min = pd.DataFrame(np.array(min_all_tasks).reshape(np.shape(min_all_tasks)[1],np.shape(min_all_tasks)[2]), columns = columns_min)

training_submission = pd.concat([submission, df_training_mean, df_training_median, df_training_max,df_training_min], axis=1)
pickle.dump(training_submission,open('save path/'+ experiment + '/Train_submission/Training_Submission2.pkl',"wb"))

################ making dataframe for test data
del temp_df, df
path = 'path of test dataframe/'+ experiment + '/'
data_type = 'Test'
temp_df = pd.read_pickle(path + 'subject1')
for sub in [1,2,3,4,5,6,7]:
    df = pd.read_pickle(path + 'subject' + str(sub+1))
    temp_df = temp_df.append(df,ignore_index = True)
d = {"dataFileHandleId": temp_df["dataFileHandleId"]}
submission = pd.DataFrame(d)
i = 0
temp_mean = []
temp_min = []
temp_max = []
temp_median = []

mean_all_tasks = []
min_all_tasks = []
max_all_tasks = []
median_all_tasks = []

for sample in range(temp_df.shape[0]):
    df = testFeatures[i:i+np.int(temp_df["sampleNum"][sample]),:]
    ####### mean ###############
    df_mean = np.mean(df,axis = 0)
    df_mean = np.transpose(df_mean[:,np.newaxis])
    ##### min ###############
    df_min = np.min(df,axis = 0 )
    df_min = np.transpose(df_min[:,np.newaxis])
    ##### max #################
    df_max = np.max(df,axis = 0)
    df_max = np.transpose(df_max[:,np.newaxis])
    ###### median #############
    df_median = np.median(df,axis = 0)
    df_median = np.transpose(df_median[:,np.newaxis])
    
    temp_mean.append(df_mean)    
    temp_min.append(df_min)
    temp_max.append(df_max)
    temp_median.append(df_median)
    
    i +=np.int(temp_df["sampleNum"][sample])
    
####### make a list of tasks #####################
mean_all_tasks.append(np.vstack(temp_mean))
min_all_tasks.append(np.vstack(temp_min))
max_all_tasks.append(np.vstack(temp_max))
median_all_tasks.append(np.vstack(temp_median))

########## assign a header for dataframe
columns_mean = ['feature'+str(j+1) for j in range(56)]
columns_median = ['feature'+str(j+1) for j in range(56,112)]
columns_max = ['feature'+str(j+1) for j in range(112,168)]
columns_min = ['feature'+str(j+1) for j in range(168,224)]

########## make a seperate dataframe per mean/max/mean/median of tasks features
df_test_mean = pd.DataFrame(np.array(mean_all_tasks).reshape(np.shape(mean_all_tasks)[1],np.shape(mean_all_tasks)[2]), columns = columns_mean)
df_test_median = pd.DataFrame(np.array(median_all_tasks).reshape(np.shape(median_all_tasks)[1],np.shape(median_all_tasks)[2]), columns = columns_median)
df_test_max = pd.DataFrame(np.array(max_all_tasks).reshape(np.shape(max_all_tasks)[1],np.shape(max_all_tasks)[2]), columns = columns_max)
df_test_min = pd.DataFrame(np.array(min_all_tasks).reshape(np.shape(min_all_tasks)[1],np.shape(min_all_tasks)[2]), columns = columns_min)


test_submission = pd.concat([submission, df_test_mean, df_test_median, df_test_max,df_test_min], axis=1)
pickle.dump(test_submission, open('save path path/'+ experiment +'/Train_submission/Test_Submission2.pkl',"wb"))

#train_submission = pickle.load(open('/Volumes/Macintosh HD 2/PhD/Third_Year/Parkinson_Challenge/LDOPA/submission/'+ experiment +'/Train_submission/Training_Submission.pkl',"rb"))

all_submission = training_submission.append(test_submission,ignore_index = True)    

all_submission.to_csv('your save path/'+experiment+'/Train_Submission/'+experiment+'2'+'.csv', index = False, sep=',')


