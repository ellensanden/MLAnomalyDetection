from matplotlib import pyplot
import keras
import tensorflow as tf
from keras.models import Model
from keras.models import Sequential
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import AveragePooling3D

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
from keras.layers import Conv3DTranspose
from keras.layers import GlobalAveragePooling2D
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
import numpy as np
from numpy import array
from tensorflow.python.keras.layers.normalization import BatchNormalization

from data_processing import process
filename = 'gear_dataset.csv'
rows = 50000   # 500  000 highest i can go without killing gpu for cnn_lstm 
data_with_attack, AttackIDs, labeled_data = process(filename,rows,no_attack_packets=False) 
print(f'including attack data: {data_with_attack.shape}')

data , IDs, _ = process(filename,rows,no_attack_packets=True) 
print(f'normal data: {data.shape}')

n_rows = data.shape[0] 
n_features = data.shape[1]

n_timesteps = 40 # 40 gave better results than 20

labeled_data = labeled_data.reset_index(drop=True)
attack = labeled_data[labeled_data['Attack'] == 'T'].copy()
attack_ind = attack.index
attack_samples = np.floor(attack_ind/n_timesteps)
attack_samples = np.unique(attack_samples) # all samples that contain attack packets
attack_samples = attack_samples.astype(int)

from prepare_data_cube import make_cubes

#type = 'timeDist_cnn'
#type = 'cnn_lstm'
type = 'cnn'
x_test,x_train,xA,last_attack_timestep = make_cubes(IDs,AttackIDs,data,data_with_attack,n_timesteps,type,labeled_data)

if type == 'cnn':
    # input = Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3], 1))

    # x = Conv3D(filters = 60, kernel_size = (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same')(input) 
    # x = MaxPool3D((2,2,2),padding='same')(x)

    # x = Conv3D(filters = 30, kernel_size = (2, 2, 2), strides=(1, 1, 1), activation='relu', padding='same')(x)
    # x = MaxPool3D((2,2,2),padding='same')(x)

    # # x = Conv3D(filters = 60, kernel_size = (2, 2, 2), strides=(1, 1, 1), activation='relu', padding='same')(x)
    # # x = UpSampling3D(size=(2,2,2))(x)

    # # x = Conv3D(filters = 1, kernel_size = (2, 2, 1), strides=(2, 2, 1), activation='relu', padding='same')(x)
    # # x = UpSampling3D(size=(2,2,2))(x)

    # # x = Conv3D(filters = 60, kernel_size = (2, 2, 1), strides=(2, 2, 1), activation='relu', padding='same')(x)
    # #x = UpSampling3D(size=(2,2,1))(x)
    # #x = UpSampling3D(size=(2,2,1))(x)
    # x = Conv3DTranspose(60,kernel_size=(2,2,2), strides=(2, 2, 2))(x)
    # x = Conv3D(filters = 1, kernel_size = (2, 2, 2), strides=(1, 1, 3), activation='relu', padding='same')(x)

    # x = Conv3DTranspose(60,kernel_size=(2,2,2), strides=(2, 2, 17))(x)
    # #x = Conv3D(filters = 30, kernel_size = (2, 2, 2), strides=(1, 6, 1), activation='relu', padding='same')(x)

    # #x = Conv3DTranspose(30,kernel_size=(2,2,2), strides=(2, 2, 2))(x)
    # x = Conv3DTranspose(20,kernel_size=(2,2,2), strides=(2, 2, 2))(x)



    # x = Conv3D(filters = 1, kernel_size = (2, 2, 2), strides=(1, 1, 4), activation='sigmoid', padding='same')(x)

    # CNN = Model(inputs=input, outputs=x,name="CNN")
    # CNN.compile(optimizer='adam', loss='mse')
    # CNN.summary()
    
    # model = CNN

    # for 20 timesteps
    
    # input = Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3], 1))

    # x = Conv3D(filters = 60, kernel_size = (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same')(input) 
    # x = MaxPool3D((2,2,2),padding='same')(x)

    # x = Conv3D(filters = 30, kernel_size = (2, 2, 2), strides=(1, 1, 1), activation='relu', padding='same')(x)
    # x = MaxPool3D((1,2,2),padding='same')(x)

    # # x = Conv3D(filters = 60, kernel_size = (2, 2, 2), strides=(1, 1, 1), activation='relu', padding='same')(x)
    # # x = UpSampling3D(size=(2,2,2))(x)

    # # x = Conv3D(filters = 1, kernel_size = (2, 2, 1), strides=(2, 2, 1), activation='relu', padding='same')(x)
    # # x = UpSampling3D(size=(2,2,2))(x)

    # # x = Conv3D(filters = 60, kernel_size = (2, 2, 1), strides=(2, 2, 1), activation='relu', padding='same')(x)
    # #x = UpSampling3D(size=(2,2,1))(x)
    # #x = UpSampling3D(size=(2,2,1))(x)
    # x = Conv3DTranspose(60,kernel_size=(1,2,2), strides=(2, 2, 2))(x)
    # x = Conv3D(filters = 1, kernel_size = (2, 2, 2), strides=(1, 1, 3), activation='relu', padding='same')(x)

    # x = Conv3DTranspose(60,kernel_size=(2,2,2), strides=(2, 2, 17))(x)
    # #x = Conv3D(filters = 30, kernel_size = (2, 2, 2), strides=(1, 6, 1), activation='relu', padding='same')(x)

    # #x = Conv3DTranspose(30,kernel_size=(2,2,2), strides=(2, 2, 2))(x)
    # x = Conv3DTranspose(20,kernel_size=(1,2,2), strides=(1, 2, 2))(x)



    # x = Conv3D(filters = 1, kernel_size = (1, 2, 2), strides=(1, 1, 4), activation='relu', padding='same')(x)

    # CNN = Model(inputs=input, outputs=x,name="CNN")
    # CNN.compile(optimizer='adam', loss='mse') # try with BinaryCrossentropy next time
    # CNN.summary()
    # model = CNN

    # added LSTM after
    # input = Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3], 1))

    # x = Conv3D(filters = 60, kernel_size = (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same')(input) 
    # x = MaxPool3D((2,2,2),padding='same')(x)

    # x = Conv3D(filters = 30, kernel_size = (2, 2, 2), strides=(1, 1, 1), activation='relu', padding='same')(x)
    # x = MaxPool3D((1,2,2),padding='same')(x)

    # # x = Conv3D(filters = 60, kernel_size = (2, 2, 2), strides=(1, 1, 1), activation='relu', padding='same')(x)
    # # x = UpSampling3D(size=(2,2,2))(x)

    # # x = Conv3D(filters = 1, kernel_size = (2, 2, 1), strides=(2, 2, 1), activation='relu', padding='same')(x)
    # # x = UpSampling3D(size=(2,2,2))(x)

    # # x = Conv3D(filters = 60, kernel_size = (2, 2, 1), strides=(2, 2, 1), activation='relu', padding='same')(x)
    # #x = UpSampling3D(size=(2,2,1))(x)
    # #x = UpSampling3D(size=(2,2,1))(x)
    # x = Conv3DTranspose(60,kernel_size=(1,2,2), strides=(2, 2, 2))(x)
    # x = Conv3D(filters = 1, kernel_size = (2, 2, 2), strides=(1, 1, 3), activation='relu', padding='same')(x)

    # x = Conv3DTranspose(60,kernel_size=(2,2,2), strides=(2, 2, 17))(x)
    # #x = Conv3D(filters = 30, kernel_size = (2, 2, 2), strides=(1, 6, 1), activation='relu', padding='same')(x)

    # #x = Conv3DTranspose(30,kernel_size=(2,2,2), strides=(2, 2, 2))(x)
    # x = Conv3DTranspose(20,kernel_size=(1,2,2), strides=(1, 2, 2))(x)



    # x = Conv3D(filters = 1, kernel_size = (1, 2, 2), strides=(1, 1, 4), activation='relu', padding='same')(x)
    # #x = tf.squeeze(x,[0,4])
    # x = tf.squeeze(x,4)
    # x = Reshape((x.shape[1]*x.shape[2],x.shape[3]))(x)
    # x = LSTM(100,return_sequences=True,return_state=False)(x) # put lstm model as a separate model after? 
    # x = LSTM(50,return_sequences=True)(x)
    # x = LSTM(17,return_sequences=True)(x)

    # x = tf.expand_dims(x,-1)
    # x = Reshape((x_train.shape[1],x_train.shape[2],x_train.shape[3], 1))(x)

    # #x = Conv3D(filters = 1, kernel_size = (1, 1, 1), strides=(1, 1, 1), activation='relu', padding='same')(x)

    # CNN = Model(inputs=input, outputs=x,name="CNN")
    # CNN.compile(optimizer='adam', loss='mse')
    # CNN.summary()

    # model = CNN

    # alternative cnn
    # input = Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3], 1))

    # x = Conv3D(filters = 10, kernel_size = (4, 4, 4), strides=(2, 2, 2), activation='relu', padding='same')(input) 
    # x = MaxPool3D((2,2,2),padding='same')(x)

    # x = Conv3D(filters = 10, kernel_size = (2, 2, 2), strides=(1, 1, 1), activation='relu', padding='same')(x)
    # x = MaxPool3D((1,2,2),padding='same')(x)

    # x = Conv3D(filters = 10, kernel_size = (2, 2, 2), strides=(1, 1, 1), activation='relu', padding='same')(x)
    # x = MaxPool3D((1,2,2),padding='same')(x)

    # # x = Conv3D(filters = 60, kernel_size = (2, 2, 2), strides=(1, 1, 1), activation='relu', padding='same')(x)
    # # x = UpSampling3D(size=(2,2,2))(x)

    # # x = Conv3D(filters = 1, kernel_size = (2, 2, 1), strides=(2, 2, 1), activation='relu', padding='same')(x)
    # # x = UpSampling3D(size=(2,2,2))(x)

    # # x = Conv3D(filters = 60, kernel_size = (2, 2, 1), strides=(2, 2, 1), activation='relu', padding='same')(x)
    # #x = UpSampling3D(size=(2,2,1))(x)
    # #x = UpSampling3D(size=(2,2,1))(x)
    # x = Conv3DTranspose(10,kernel_size=(1,2,2), strides=(2, 2, 2))(x)
    # x = Conv3D(filters = 1, kernel_size = (2, 2, 2), strides=(1, 1, 3), activation='relu', padding='same')(x)

    # x = Conv3DTranspose(20,kernel_size=(2,2,2), strides=(2, 2, 17))(x)
    # #x = Conv3D(filters = 30, kernel_size = (2, 2, 2), strides=(1, 6, 1), activation='relu', padding='same')(x)

    # #x = Conv3DTranspose(30,kernel_size=(2,2,2), strides=(2, 2, 2))(x)
    # x = Conv3DTranspose(20,kernel_size=(1,2,2), strides=(1, 2, 2))(x)
    # x = Conv3DTranspose(20,kernel_size=(1,1,1), strides=(1, 2, 1))(x)


    # x = Conv3D(filters = 1, kernel_size = (2, 2, 2), strides=(1, 1, 4), activation='sigmoid', padding='same')(x)
    # #x = tf.squeeze(x,[0,4])
    # # x = tf.squeeze(x,4)
    # # x = Reshape((x.shape[1]*x.shape[2],x.shape[3]))(x)
    # # x = LSTM(100,return_sequences=True,return_state=False)(x) # put lstm model as a separate model after? 
    # # x = LSTM(50,return_sequences=True)(x)
    # # x = LSTM(17,return_sequences=True)(x)

    # # x = tf.expand_dims(x,-1)
    # # x = Reshape((x_train.shape[1],x_train.shape[2],x_train.shape[3], 1))(x)

    # #x = Conv3D(filters = 1, kernel_size = (1, 1, 1), strides=(1, 1, 1), activation='relu', padding='same')(x)

    # CNN = Model(inputs=input, outputs=x,name="CNN")
    # CNN.compile(optimizer='adam', loss='mse')
    # CNN.summary()

    # model = CNN
    # alternative
    # input = Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3], 1))

    # x = Conv3D(filters = 60, kernel_size = (11, 11, 11), strides=(1, 1, 1), activation='relu', padding='same')(input) 
    # x = MaxPool3D((2,2,2),padding='same')(x)

    # x = Conv3D(filters = 10, kernel_size = (9, 9, 3), strides=(1, 1, 1), activation='relu', padding='same')(x)
    # x = MaxPool3D((1,2,2),padding='same')(x)

    # x = Conv3D(filters = 5, kernel_size = (3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(x)
    # x = MaxPool3D((2,2,2),padding='same')(x)

    # # x = Conv3D(filters = 60, kernel_size = (2, 2, 2), strides=(1, 1, 1), activation='relu', padding='same')(x)
    # # x = UpSampling3D(size=(2,2,2))(x)

    # # x = Conv3D(filters = 1, kernel_size = (2, 2, 1), strides=(2, 2, 1), activation='relu', padding='same')(x)
    # # x = UpSampling3D(size=(2,2,2))(x)

    # # x = Conv3D(filters = 60, kernel_size = (2, 2, 1), strides=(2, 2, 1), activation='relu', padding='same')(x)
    # #x = UpSampling3D(size=(2,2,1))(x)
    # #x = UpSampling3D(size=(2,2,1))(x)
    # x = Conv3DTranspose(5,kernel_size=(3,3,3), strides=(1, 1, 1))(x)
    # #x = Conv3D(filters = 10, kernel_size = (9, 9, 2), strides=(1, 1, 3), activation='relu', padding='same')(x)

    # x = Conv3DTranspose(10,kernel_size=(9,9,3), strides=(1, 1, 1))(x)
    # #x = Conv3D(filters = 30, kernel_size = (2, 2, 2), strides=(1, 6, 1), activation='relu', padding='same')(x)

    # #x = Conv3DTranspose(30,kernel_size=(2,2,2), strides=(2, 2, 2))(x)
    # x = Conv3DTranspose(10,kernel_size=(11,11,11), strides=(1, 1, 1))(x)
    # x = Conv3DTranspose(1,kernel_size=(11,37,1), strides=(1, 1, 1))(x)


    # #x = Conv3D(filters = 1, kernel_size = (2, 2, 2), strides=(1, 1, 4), activation='sigmoid', padding='same')(x)
    # #x = tf.squeeze(x,[0,4])
    # # x = tf.squeeze(x,4)
    # # x = Reshape((x.shape[1]*x.shape[2],x.shape[3]))(x)
    # # x = LSTM(100,return_sequences=True,return_state=False)(x) # put lstm model as a separate model after? 
    # # x = LSTM(50,return_sequences=True)(x)
    # # x = LSTM(17,return_sequences=True)(x)

    # # x = tf.expand_dims(x,-1)
    # # x = Reshape((x_train.shape[1],x_train.shape[2],x_train.shape[3], 1))(x)

    # #x = Conv3D(filters = 1, kernel_size = (1, 1, 1), strides=(1, 1, 1), activation='relu', padding='same')(x)

    # CNN = Model(inputs=input, outputs=x,name="CNN")
    # CNN.compile(optimizer='adam', loss='BinaryCrossentropy')
    # CNN.summary()

    # model = CNN

    # new alt with even larger kernels
    # input = Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3], 1))

    # x = Conv3D(filters = 60, kernel_size = (21, 21, 21), strides=(1, 1, 1), activation='relu', padding='same')(input) 
    # x = MaxPool3D((2,2,2),padding='same')(x)

    # x = Conv3D(filters = 30, kernel_size = (15, 15, 15), strides=(1, 1, 1), activation='relu', padding='same')(x) 
    # x = MaxPool3D((2,2,2),padding='same')(x)

    # x = Conv3D(filters = 20, kernel_size = (11, 11, 11), strides=(1, 1, 1), activation='relu', padding='same')(x) 
    # x = MaxPool3D((2,2,2),padding='same')(x)

    # x = Conv3D(filters = 10, kernel_size = (9, 9, 3), strides=(1, 1, 1), activation='relu', padding='same')(x)
    # x = MaxPool3D((1,2,2),padding='same')(x)

    # #x = Conv3D(filters = 5, kernel_size = (3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(x)
    # #x = MaxPool3D((2,2,2),padding='same')(x)

    # x = Conv3DTranspose(5,kernel_size=(3,3,3), strides=(1, 1, 1))(x)

    # x = Conv3DTranspose(1,kernel_size=(9,9,3), strides=(1, 1, 1))(x)

    # x = Conv3DTranspose(1,kernel_size=(11,11,11), strides=(1, 1, 1))(x)

    # k1 = x_train.shape[1] - x.shape[1] + 1
    # k2 = x_train.shape[2] - x.shape[2] + 1
    # k3 = x_train.shape[3] - x.shape[3] + 1 

    # x = Conv3DTranspose(1,kernel_size=(k1,k2,k3), strides=(1, 1, 1))(x) # this should work always as long as strides are 1,1,1

    # CNN = Model(inputs=input, outputs=x,name="CNN")
    # CNN.compile(optimizer='adam', loss='BinaryCrossentropy')
    # CNN.summary()

    # model = CNN

    # another alt
    input = Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3], 1))

    x = Conv3D(filters = 5, kernel_size = (40, 40, 40), strides=(1, 1, 1), activation='relu', padding='same')(input) 
    x = MaxPool3D((2,2,2),padding='same')(x)

    x = Conv3D(filters = 5, kernel_size = (21, 21, 21), activation='relu', strides=(1, 1, 1), padding='same')(x) 
    x = MaxPool3D((2,2,2),padding='same')(x)

    x = Conv3D(filters = 3, kernel_size = (15, 15, 15),  activation='relu', strides=(1, 1, 1), padding='same')(x) 
    x = MaxPool3D((2,2,2),padding='same')(x)

    x = Conv3D(filters = 3, kernel_size = (11, 11, 11), strides=(1, 1, 1), activation='relu', padding='same')(x) 
    x = MaxPool3D((2,2,2),padding='same')(x)

    x = Conv3D(filters = 2, kernel_size = (9, 9, 3), strides=(1, 1, 1), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool3D((1,2,2),padding='same')(x)

    #x = Conv3D(filters = 5, kernel_size = (3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(x)
    #x = MaxPool3D((2,2,2),padding='same')(x)

    x = Conv3DTranspose(2,kernel_size=(3,3,3), strides=(1, 1, 1))(x)

    x = Conv3DTranspose(1,kernel_size=(9,9,3), strides=(1, 1, 1))(x)

    x = Conv3DTranspose(1,kernel_size=(11,11,11), strides=(1, 1, 1))(x)

    k1 = x_train.shape[1] - x.shape[1] + 1
    k2 = x_train.shape[2] - x.shape[2] + 1
    k3 = x_train.shape[3] - x.shape[3] + 1 

    x = Conv3DTranspose(1,kernel_size=(k1,k2,k3), strides=(1, 1, 1))(x) # this should work always as long as strides are 1,1,1

    CNN = Model(inputs=input, outputs=x,name="CNN")
    CNN.compile(optimizer='adam', loss='BinaryCrossentropy')
    CNN.summary()

    model = CNN
    # without batchnorm: 0.5028
    # with batch norm in middle: es 0.4705 (maxpool,relu)
    # with batch norm everywhere: worse than just in middle
    # with batch norm in middle: es 2.48 (avgpool,relu)
    # with batch norm in middle: 0.4722  (maxpool, sigmoid) 
    #^on only conv3d, not transpose
    # on first transp. below
     # with batch norm in middle: 0.4784  (maxpool, softmax)
if type == 'cnn_lstm':
    # input = Input(shape=(n_features,2,n_timesteps))

    # x = Conv2D(filters = 60, kernel_size = (4, 1), activation='relu', padding='same')(input) 
    # x = MaxPool2D((2,1),padding='valid')(x)

    # x = Conv2D(filters = n_timesteps, kernel_size = (1, 1), activation='relu', padding='same')(x)
    # x = MaxPool2D((2, 1))(x)

    # x = Conv2D(filters = n_timesteps, kernel_size = (2, 1), activation='relu', padding='same')(x)
    # x = MaxPool2D((2, 2))(x)

    # x = Reshape((n_timesteps,8, ))(x)

    # x = LSTM(64,return_sequences = True)(x)
    # x = LSTM(128, return_sequences = True)(x)

    # x = Reshape((64,2,n_timesteps,))(x)
    # x = Conv2D(filters = n_timesteps, kernel_size = (2, 1), activation='relu', padding='same')(x)
    # CNN = Model(inputs=input, outputs=x,name="CNN")
    # CNN.compile(optimizer='adam', loss='KLDivergence')
    # CNN.summary()
    # model = CNN
    input = Input(shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3]))

    x = Conv2D(filters = 60, kernel_size = (4, 4), activation='relu', padding='same')(input) 
    x = MaxPool2D((2,2),padding='valid')(x)

    x = Conv2D(filters = n_timesteps, kernel_size = (2, 2), activation='relu', padding='same')(x)
    x = MaxPool2D((2, 2))(x)

    x = Reshape((n_timesteps,64))(x)
    # LSTM [samples,timesteps,features]
    x = LSTM(64,return_sequences = True)(x)
    x = LSTM(128, return_sequences = True)(x)

    x = Reshape((64,2,n_timesteps,))(x)
    x = Conv2D(filters = 20, kernel_size = (1, 1), activation='relu', padding='same')(x)
    x =  UpSampling2D((1, 17))(x)
    x = Reshape((64,17,n_timesteps))(x)



    CNN = Model(inputs=input, outputs=x,name="CNN")
    CNN.compile(optimizer='adam', loss='BinaryCrossentropy')
    CNN.summary()

    model = CNN


if type == 'timeDist_cnn':
    
    model = Sequential()
    n_features = 64
    model.add(
        TimeDistributed(
            Conv2D(60, (2,1), activation='relu',padding='same'), 
            input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[3]
    , 1) # 5 images...
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
            UpSampling2D((2,17))
        )
    )

    model.add(
        TimeDistributed(
            Conv2D(60, (1,1), activation='relu',padding='same')
        )
    )

    model.compile('adam', loss='BinaryCrossentropy')

    model.summary()

import time

es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)

s = time.time()
# could probably shuffle for 3d cnn
history = model.fit(x_train,x_train, validation_data=(x_test, x_test), epochs=5000, verbose=2, shuffle=True, batch_size = 40, callbacks = [es])

e = time.time()


print(f'training time = {e-s} seconds')

model.save('CNN_LSTM_monday')