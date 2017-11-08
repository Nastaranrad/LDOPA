
"""
@author: Nastaran Mohammadian Rad, nastaran.mrad@gmail.com 
############# Feature learning via CNN, model evaluation
keras v.'2.0.8'
tensorflow v.'1.2.1'
python v.'2.7'
"""
import numpy as np
import scipy.io as sio
from keras.layers import Conv1D, AveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.callbacks  import EarlyStopping
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_curve, roc_auc_score, auc, precision_recall_curve, average_precision_score
from keras.models import Sequential
from keras.layers.core import Activation, Dense, Flatten, Dropout
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop
from sklearn.preprocessing import StandardScaler
from keras import backend as K
import pickle
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

# Setting Parameters
nb_filters = [4, 4, 8]
kernel_size = 5
hidden_neuron_num = 16
pool_size = 3
stride_size = 2
channels = 3
runNum = 10
experiment = 'Tremor'
nb_epoch = [10, 5]  
if experiment == 'Tremor':
    learn_rate = 0.008 
    nb_classes = 5

else:
    learn_rate = 0.001
    nb_classes = 2
batch_size = 100
mtm = 0.9
padding = 'same'

f1Net = np.zeros([runNum,nb_classes])
precisionNet = np.zeros([runNum,nb_classes])
recallNet = np.zeros([runNum,nb_classes])
accNet = np.zeros([runNum,nb_classes])
AUCNet = np.zeros([runNum,nb_classes])
prcNet = np.zeros([runNum,nb_classes])
average_precision = np.zeros([runNum,nb_classes])

fpr = dict()
tpr = dict()
roc_auc = dict()

path = 'path of saved data/'+ experiment+ '/Raw_data/'
savePath = 'your save path/'

# Loading data
matContent = sio.loadmat(path + experiment + '_raw_data.mat')
trainingFeatures = matContent['trainingFeatures']
testFeatures = matContent['testFeatures']
trainingLabels = matContent['trainingLabels']
del matContent

trainingLabels = trainingLabels.astype(int)
trainingLabels = np.squeeze(trainingLabels)
trainingLabels = np_utils.to_categorical(trainingLabels, nb_classes)

# Normalization
scaler = StandardScaler()
scaler.fit(np.reshape(trainingFeatures, [trainingFeatures.shape[0], trainingFeatures.shape[1]*trainingFeatures.shape[2]]))
trainingFeatures = scaler.transform(np.reshape(trainingFeatures, [trainingFeatures.shape[0], trainingFeatures.shape[1]*trainingFeatures.shape[2]]))
trainingFeatures = np.reshape(trainingFeatures, [trainingFeatures.shape[0],testFeatures.shape[1],testFeatures.shape[2]])
testFeatures = scaler.transform(np.reshape(testFeatures, [testFeatures.shape[0], testFeatures.shape[1]*testFeatures.shape[2]]))
testFeatures = np.reshape(testFeatures,[testFeatures.shape[0], trainingFeatures.shape[1], trainingFeatures.shape[2]])

#### train network
for run in range(runNum):
    model = Sequential()
    model.add(Conv1D(filters=nb_filters[0], kernel_size=kernel_size, padding=padding, activation='relu',
                     input_shape=(trainingFeatures.shape[1], trainingFeatures.shape[2])))
    model.add(AveragePooling1D(pool_size=pool_size, strides=stride_size, padding=padding))
    model.add(Conv1D(filters=nb_filters[1], kernel_size=kernel_size, padding=padding, activation='relu',
                     kernel_initializer='he_normal'))
    model.add(AveragePooling1D(pool_size=pool_size, strides=stride_size, padding=padding))
    model.add(Conv1D(filters=nb_filters[2], kernel_size=kernel_size, padding=padding, activation='relu',
                     kernel_initializer='he_normal'))
    model.add(AveragePooling1D(pool_size=pool_size, strides=stride_size, padding=padding))
    model.add(Flatten())

    model.add(Dense(8, kernel_initializer='he_normal', activation='relu'))
    model.add(Dense(5))
    model.add(Activation('softmax'))

    earlyStopping = EarlyStopping(monitor='val_loss', patience=2, verbose=1, mode='auto')

    for ep in range(len(nb_epoch)):
        if experiment == 'Tremor':
            optimizer = RMSprop(lr=learn_rate / 10 ** ep)
        else:
            optimizer = SGD(lr=learn_rate / 10 ** ep)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        model.fit(trainingFeatures, trainingLabels, batch_size=batch_size, epochs=nb_epoch[ep],
                  verbose=2, callbacks=[earlyStopping], validation_split=0.2)
        
    ##################### save learned features    
    get_3rd_layer_output = K.function([model.layers[0].input], [model.layers[6].output,]) 
    layer_output_training = get_3rd_layer_output([trainingFeatures])[0] 
    layer_output_test = get_3rd_layer_output([testFeatures])[0]
    sio.savemat(savePath + experiment + '_Learned_Features' +'_run_'+ str(run+1)+ '.mat', {'trainingFeatures':layer_output_training, 'trainingLabels':trainingLabels,
        'testFeatures':layer_output_test})
    
    ############## model prediction and evaluation ##############
    if experiment== 'Tremor':          
        predicted_trainingLabels = model.predict_classes(trainingFeatures,verbose = 0)
        predicted_trainingLabels = np_utils.to_categorical(predicted_trainingLabels, nb_classes)    
        soft_targets_training = model.predict(trainingFeatures,verbose = 0)        
        for i in range(nb_classes):
            fpr[run, i], tpr[run, i], _ = roc_curve(trainingLabels[:,i], soft_targets_training[:, i])
            roc_auc[run, i] = auc(fpr[run, i], tpr[run, i])
            precisionNet[run, i] = precision_score(trainingLabels[:,i], predicted_trainingLabels[:, i])
            recallNet[run, i] = recall_score(trainingLabels[:,i], predicted_trainingLabels[:, i])
            accNet[run,i] = accuracy_score(trainingLabels[:,i], predicted_trainingLabels[:, i])
            f1Net[run,i] = f1_score(trainingLabels[:,i], predicted_trainingLabels[:, i])
            print('Run %d :i %d :precisionNet: %.4f' % ( run + 1,i, precisionNet[run,i]))
            AUCNet[run,i] = roc_auc_score(trainingLabels[:,i], soft_targets_training[:,i])
            print('Run %d:i %d :AUC_Net: %.4f' % (run+1, i, AUCNet[run,i]))        
    else:
        predicted_trainingLabels = model.predict_classes(trainingFeatures,verbose = 0)
        soft_targets_training = model.predict(trainingFeatures,verbose = 0)        
        fpr[run, 0], tpr[run, 0], _ = roc_curve(trainingLabels[:,1], soft_targets_training[:, 1])
        roc_auc[run, 0] = auc(fpr[run, 0], tpr[run, 0])
        precisionNet[run, 0] = precision_score(trainingLabels[:,1], predicted_trainingLabels[:, 1])
        recallNet[run, 0] = recall_score(trainingLabels[:,1], predicted_trainingLabels[:, 1])
        accNet[run,0] = accuracy_score(trainingLabels[:,1], predicted_trainingLabels[:, 1])
        f1Net[run,0] = f1_score(trainingLabels[:,i], predicted_trainingLabels[:, 1])
        print('Run %d :precisionNet: %.4f' % ( run + 1, precisionNet[run,0]))
        AUCNet[run,0] = roc_auc_score(trainingLabels[:,1], soft_targets_training[:,1])
        print('Run %d:AUC_Net: %.4f' % (run+1, AUCNet[run,0]))        
        
    # save the model and weights
    json_string = model.to_json()
    open(savePath + experiment + '_CNN' + '_Run_' + str(run + 1) + '.json', 'w').write(json_string)
    model.save_weights(savePath + experiment + '_CNN_Run_' + str(run + 1) + '.h5', overwrite=True)
    
    #save results
    sio.savemat(savePath + experiment + '_CNN_Results' + '.mat', {'precisionNet': precisionNet,
                                                                'recallNet': recallNet, 'f1Net': f1Net,
                                                                'accNet': accNet,'AUCNet':AUCNet})

pickle.dump( fpr, open(savePath + 'CNN_pickle_fpr.p' , "wb" ) )
pickle.dump( tpr, open(savePath + 'CNN_pickle_tpr.p', "wb" ) )
pickle.dump( roc_auc, open(savePath + 'CNN_pickle_roc.p', "wb" ) ) 
