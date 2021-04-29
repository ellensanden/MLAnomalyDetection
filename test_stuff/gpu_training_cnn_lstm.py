from matplotlib import pyplot
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
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
import numpy as np
from numpy import array

from data_processing import process
filename = 'gear_dataset.csv'
rows = 50000  # no attack data in the first 1000 rows
data_with_attack, AttackIDs = process(filename,rows,no_attack_packets=False) # change name of attackIDs..
print(f'including attack data: {data_with_attack.shape}')

data , IDs = process(filename,rows,no_attack_packets=True)
print(f'normal data: {data.shape}')

n_rows = data.shape[0] 
n_features = data.shape[1]


def convert_from_hex(hex,output_type): # converts the data in hex from hexadecimal to decimal or binary form
     out = np.zeros((hex.size))
     if output_type == 'dec':
        for x in range(hex.size):
            h_value = hex[x]
            out[x] = int(h_value,16)
     else:
        for x in range(hex.size):
            h_value = hex[x]
            binary[x] = bin(int(h_value, 16))[2:]

     return out


#data = convert_from_hex(Data,'dec')
IDs = IDs.reset_index(drop=True)

id = convert_from_hex(IDs,'dec') 

ID_matrix =  array([[id],]*data.shape[1]).transpose()
ID_matrix = np.squeeze(ID_matrix)
dataCube = np.dstack([data,ID_matrix])
AttackIDs = AttackIDs.reset_index(drop=True)

Attackid = convert_from_hex(AttackIDs,'dec') 

ID_matrixA =  array([[Attackid],]*data_with_attack.shape[1]).transpose()
ID_matrixA = np.squeeze(ID_matrixA)
dataCubeA = np.dstack([data_with_attack,ID_matrixA])

n_timesteps = 12
n_samples = int(np.floor(dataCube.shape[0]/n_timesteps))
print(n_samples)

last_timestep = n_samples*n_timesteps
x = dataCube[0:last_timestep,:,:]
print(x.shape)
x = x.reshape(n_samples,64,2,n_timesteps)
print(x.shape)

train_size = int(np.floor(0.7*n_samples))
x_train = x[0:train_size,:,:,:]
x_test = x[train_size:,:,:,:]

print(x_test.shape, x_train.shape)


n_samples = int(np.floor(dataCubeA.shape[0]/n_timesteps))
print(n_samples)

last_timestep = n_samples*n_timesteps
xA = dataCubeA[0:last_timestep,:,:]
print(xA.shape)
xA = xA.reshape(n_samples,64,2,n_timesteps)
print(xA.shape)

model =  keras.models.load_model('CNN_LSTM_trained_on_50000_nonattack')

import time

es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)

s = time.time()

history = model.fit(x_train,x_train, validation_data=(x_test, x_test), epochs=300, verbose=2, shuffle=False, callbacks = [es])

e = time.time()

# plot history
pyplot.plot(history.history['loss'], label = 'train')
pyplot.plot(history.history['val_loss'], label = 'validation')

pyplot.legend()
pyplot.show()
print(f'training time = {e-s} seconds')