# get data
from matplotlib import pyplot
import pickle
import keras
import tensorflow as tf
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import TimeDistributed 
from keras.layers import RepeatVector
from keras.layers import GRU
from keras.layers import Conv3D
from keras.layers import UpSampling3D
from keras.layers import MaxPool3D
from keras.layers import Conv2D
from keras.layers import UpSampling2D
from keras.layers import MaxPool2D
from keras.layers import Flatten
from keras.layers import Reshape
from keras.layers import GlobalAveragePooling2D
from keras.layers import Conv2DTranspose
from keras.layers import Conv3DTranspose

from keras.callbacks import EarlyStopping
import numpy as np
from numpy import array
import pandas as pd

# OWN DATA
# nRows = 50000
n_timesteps = 40

# filename = 'normal_and_attack_data.csv'
    
# df = pd.read_csv(filename, nrows = nRows, sep=',')

# #bin column names
# bin_data_column_names = []
# for j in range(64):
#     bin_data_column_names.append('bin' + str(j))
    
# # Få binärvärderna av de första 5 datapunkterna i en dataframe
# temporary_df = df[bin_data_column_names]

# # The values of these binary arrays
# temporary_df_values = temporary_df.values

# data = np.array(temporary_df_values)
# data_with_attack = data.copy()
# data = data[df['Attack'] == 'No']

# IDs = df['ID']
# AttackIDs = IDs.copy()
# IDs = IDs[df['Attack'] == 'No']

# labeled_data = df.copy()

#GEAR DATA

from data_processing import process
filename = 'gear_dataset.csv'
rows = 'slice' 
data_with_attack, AttackIDs, labeled_data = process(filename,rows,no_attack_packets=False) 
print(f'including attack data: {data_with_attack.shape}')

data , IDs, _ = process(filename,rows,no_attack_packets=True)
print(f'normal data: {data.shape}')

n_rows = data.shape[0] 
n_features = data.shape[1]


labeled_data = labeled_data.reset_index(drop=True)

from prepare_data_cube import make_cubes

#type = 'timeDist_cnn'
#type = 'cnn_lstm'
type = 'cnn'

x_test,x_train,xA,lastA,samples = make_cubes(IDs,AttackIDs,data,data_with_attack,n_timesteps,type)

# attack = labeled_data[labeled_data['Attack'] == 'T'].copy()
# attack_ind = attack.index
# attack_ind = attack_ind[attack_ind<lastA] # why not <=
# attack_samples = np.floor(attack_ind/n_timesteps)
# attack_samples = np.unique( attack_samples) # all samples that contain attack packets
# attack_samples = attack_samples.astype(int)
attack = labeled_data[labeled_data['Attack'] == 'T'].copy()
#attack = labeled_data[labeled_data['Attack'] == 'Yes'].copy()
attack_ind = attack.index
attack_ind = attack_ind[attack_ind<=lastA]

contains_attack = [np.any(np.in1d(x, attack_ind)) for x in samples]
attack_samples = contains_attack
print(f'number of attack samples {len(attack_samples)}')
# run net and save parameters
modelname = 'checkpoint'
#modelname = '3dCNN_05-18_trained_on_50000_r'
CNN =  keras.models.load_model(modelname)

if type == 'timeDist_cnn' or type == 'cnn':
    attack_cubes = xA[attack_samples,:,:,:,:]

if type == 'cnn_lstm':
    attack_cubes = xA[attack_samples,:,:,:]

x_test = x_train
print(x_test.shape)
yHat_normal = CNN.predict(x_test) # only normal packets
normal_errors = x_test-yHat_normal
print(yHat_normal.shape)
normal_errors = normal_errors.flatten()
normal_errors = np.abs(normal_errors)
normal_errors = np.mean(normal_errors)
print(f'normal: {normal_errors}')



yHat_true_attack = CNN.predict(attack_cubes) # only attack packets
true_attack_errors = attack_cubes-yHat_true_attack
print(yHat_true_attack.shape)
true_attack_errors = true_attack_errors.flatten()
true_attack_errors = np.abs(true_attack_errors)
true_attack_errors = np.mean(true_attack_errors)
print(f'only attack: {true_attack_errors}')


# get average for each sample cube 
normal_errors = x_test-yHat_normal

if type == 'timeDist_cnn' or type == 'cnn':
    normal_errors = normal_errors.reshape(normal_errors.shape[0],normal_errors.shape[1]*normal_errors.shape[2]*normal_errors.shape[3]*normal_errors.shape[4])

if type == 'cnn_lstm':
    normal_errors = normal_errors.reshape(normal_errors.shape[0],normal_errors.shape[1]*normal_errors.shape[2]*normal_errors.shape[3])

normal_errors = np.abs(normal_errors)
normal_errors = np.sum(normal_errors,axis=1)
print(normal_errors.shape)

true_attack_errors = attack_cubes-yHat_true_attack
if type == 'timeDist_cnn' or type =='cnn':
    true_attack_errors = true_attack_errors.reshape(true_attack_errors.shape[0],true_attack_errors.shape[1]*true_attack_errors.shape[2]*true_attack_errors.shape[3] *true_attack_errors.shape[4])

if type == 'cnn_lstm':
    true_attack_errors = true_attack_errors.reshape(true_attack_errors.shape[0],true_attack_errors.shape[1]*true_attack_errors.shape[2]*true_attack_errors.shape[3])

true_attack_errors = np.abs(true_attack_errors)
true_attack_errors = np.sum(true_attack_errors,axis=1)
print(true_attack_errors.shape)

#maxNormal = np.max(normal_errors) + 0.01*np.max(normal_errors)
#minAttack = np.min(true_attack_errors) - 0.01*np.max(normal_errors)

# save normal errors and attack errors

f = open('normal_errors.pckl', 'wb')
pickle.dump(normal_errors, f)
f.close()


f = open('attack_errors.pckl', 'wb')
pickle.dump(true_attack_errors, f)
f.close()