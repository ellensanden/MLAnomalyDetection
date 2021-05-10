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
from keras.layers import GlobalAveragePooling2D
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
import numpy as np
from numpy import array

from data_processing import process
filename = 'gear_dataset.csv'
rows = 100000  # no attack data in the first 1000 rows
data_with_attack, AttackIDs, labeled_data = process(filename,rows,no_attack_packets=False) 
print(f'including attack data: {data_with_attack.shape}')

data , IDs, _ = process(filename,rows,no_attack_packets=True)
print(f'normal data: {data.shape}')

n_rows = data.shape[0] 
n_features = data.shape[1]

n_timesteps = 40

labeled_data = labeled_data.reset_index(drop=True)
attack = labeled_data[labeled_data['Attack'] == 'T'].copy()
attack_ind = attack.index
attack_samples = np.floor(attack_ind/n_timesteps)
attack_samples = np.unique(attack_samples) # all samples that contain attack packets
attack_samples = attack_samples.astype(int)

from prepare_data_cube import make_cubes

type = 'timeDist_cnn'
#type = 'cnn_lstm'
x_test,x_train,xA = make_cubes(IDs,AttackIDs,data,data_with_attack,n_timesteps,type,labeled_data)

if type == 'cnn_lstm':
    input = Input(shape=(n_features,2,n_timesteps))

    x = Conv2D(filters = 60, kernel_size = (4, 1), activation='relu', padding='same')(input) 
    x = MaxPool2D((2,1),padding='valid')(x)

    x = Conv2D(filters = n_timesteps, kernel_size = (1, 1), activation='relu', padding='same')(x)
    x = MaxPool2D((2, 1))(x)

    x = Conv2D(filters = n_timesteps, kernel_size = (2, 1), activation='relu', padding='same')(x)
    x = MaxPool2D((2, 2))(x)

    x = Reshape((n_timesteps,8, ))(x)

    x = LSTM(64,return_sequences = True)(x)
    x = LSTM(128, return_sequences = True)(x)

    x = Reshape((64,2,n_timesteps,))(x)
    x = Conv2D(filters = n_timesteps, kernel_size = (2, 1), activation='relu', padding='same')(x)
    CNN = Model(inputs=input, outputs=x,name="CNN")
    CNN.compile(optimizer='adam', loss='KLDivergence')
    CNN.summary()
    model = CNN


if type == 'timeDist_cnn':
    
    model = Sequential()
    n_features = 64
    model.add(
        TimeDistributed(
            Conv2D(60, (2,1), activation='relu',padding='same'), 
            input_shape=(n_timesteps, n_features, 2, 1) # 5 images...
        )
    )
    model.add(
        TimeDistributed(
            Conv2D(30, (2,1), activation='relu',padding='same')
        )
    )

    model.add(
        TimeDistributed(
            MaxPool2D((2, 2),padding='same')
        )
    )

    model.add(
        Reshape((n_timesteps,-1)))

    model.add(
    LSTM(128, activation='relu', return_sequences=True)
    )

    model.add(
    LSTM(10, activation='relu', return_sequences=True)
    )

    model.add(
    LSTM(128, activation='relu', return_sequences=True)
    )

    model.add(
        TimeDistributed(
            Dense(960)
        )
    )

    model.add(
        Reshape((n_timesteps,32,1,30)))

    model.add(
        TimeDistributed(
            UpSampling2D((2,2))
        )
    )

    model.add(
        TimeDistributed(
            Conv2D(60, (1,1), activation='relu',padding='same')
        )
    )

    model.compile('adam', loss='KLDivergence')

    model.summary()

import time

es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)

s = time.time()

history = model.fit(x_train,x_train, validation_data=(x_test, x_test), epochs=1000, verbose=2, shuffle=False, callbacks = [es])

e = time.time()


print(f'training time = {e-s} seconds')

model.save('CNN_LSTM_monday')