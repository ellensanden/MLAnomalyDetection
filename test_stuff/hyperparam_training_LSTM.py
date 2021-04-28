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
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
import numpy as np
from numpy import array
import pandas as pd

colnames = ["time", "ID", "DLC", "Data1", \
        "Data2", "Data3", "Data4", "Data5", "Data6", "Data7", "Data8", "Attack"]

#nRows = 100000 #number of rows that you want
#nRows = 4443142
df = pd.read_csv(datafile, sep=',', names=colnames)
#df = pd.read_csv('gear_dataset.csv', sep=',', names=colnames)

uniqueIDs = df['ID'].unique() #26 for the entire dataset

#Drop attack packets
attack = df[df['Attack'] == 'T'].copy()
df.drop(attack.index, axis=0, inplace=True)

#Drop DLC = 2 packets
dlc2 = df[df['DLC'] == 2]
df.drop(dlc2.index, axis=0, inplace=True) #drop all dlc2 indexes

#Reset index from 1 to n (not needed actually, so commenting out)
#df.set_index(np.arange(len(df)), inplace=True)

#Pick an ID
#id_data= df[df['ID'] == '0140'].copy()
id_data = df # to use all ids
#Just use data values without time, Attack, ID and DLC right now
dataValues = id_data.drop(["time", "Attack", "ID", "DLC"], axis = 1).copy()
#dataValues.to_csv (r'one_id.csv', index=None)

dataValues = dataValues.to_numpy()


storage = np.zeros((len(dataValues),64), dtype=int)
for currentRow in np.arange(len(storage)):
    
    tempString = "".join(dataValues[currentRow])
    formatted = format(int(tempString, base=16), "064b")
    storage[currentRow,:] = np.array(list(formatted), dtype=int)

data = storage
n_rows = data.shape[0] 
n_features = data.shape[1]



def overlap_window (window_size,overlap,seq): # overlap 1 is max. larger number would be less overlap
    import numpy as np
    from numpy import array
    seq = array([seq[i : i + window_size] for i in range(0, len(seq), overlap)]) 
   
    correct = [len(x)==window_size for x in seq]
    seq = seq[correct]
    seq = np.stack(seq, axis=0 )
    seq = seq.reshape(-1,window_size,1)

    return seq

time_steps = 40
a = np.r_[0:n_rows]

X_train_samples = overlap_window(time_steps,20,a)
X_train = data[X_train_samples,:]
X_train = np.squeeze(X_train)
# print(X_train.shape)

X_reversed = np.flip(X_train,1)

def model_builder(hp):
   # do the lstm layers need to have the same number of units? will do that
   # do the two dense layers need to have the same number of units?
    n_samples = X_train.shape[0]
    time_steps = X_train.shape[1]
    n_features = X_train.shape[2]


    uniformInitializerMin = hp.Float('uniformMin', min_value=-1, max_value=0,step=0.01)
    uniformInitializerMax = hp.Float('uniformMax', min_value=0, max_value=1,step=0.01)
    lstm_initializer = tf.keras.initializers.RandomUniform(minval=uniformInitializerMin, maxval=uniformInitializerMax) 
    
    lstm_units = hp.Int('lstmUnits', min_value=1, max_value=500, step=10)

    # define Encoder
    EncoderInputs = Input(shape=(time_steps,n_features))
    
    dense_units = hp.Int('denseUnits', min_value=10, max_value=5000, step=50)
    dense1 =Dense(units=dense_units, activation='tanh')(EncoderInputs)
    
    dropoutP = hp.Float('dropout',min_value=0.0, max_value=0.95, step=0.05)
    dropout = Dropout(dropoutP)(dense1) 
  
    lstm1,state_h,state_c = LSTM(units=lstm_units,kernel_initializer =lstm_initializer, recurrent_initializer=lstm_initializer, return_state = True)(dropout)
    lstm2, state_h,state_c = LSTM(units=lstm_units,kernel_initializer =lstm_initializer, recurrent_initializer=lstm_initializer, return_state = True)(dropout,initial_state=[state_h,state_c])

    #decoder
    lstm3, state_h, state_c = LSTM(units=lstm_units,kernel_initializer =lstm_initializer, recurrent_initializer=lstm_initializer, return_state = True)(dropout,initial_state=[state_h,state_c])
    lstm4 = LSTM(units=lstm_units,kernel_initializer =lstm_initializer, recurrent_initializer=lstm_initializer, return_sequences = True)(dropout,initial_state=[state_h,state_c])

    
    output = Dense(n_features,activation= 'sigmoid')(lstm4)
    
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    EncoderDecoder = Model(inputs=EncoderInputs, outputs=output,name="EncoderDecoder")
    EncoderDecoder.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), loss='binary_crossentropy')#, metrics=[tf.keras.metrics.BinaryCrossentropy()])

    return EncoderDecoder

import kerastuner as kt
tuner = kt.Hyperband(model_builder,
                     overwrite=True,
                     objective='val_loss',
                     max_epochs=100,
                     factor=3,
                     directory='C:\ tune',
                     #directory='tuning',
                     project_name='LSTM' 
                     )

es= EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)
tuner.search(X_train, X_train, epochs=1000, validation_split=0.2, callbacks=[es])

# Get the optimal hyperparameters
best_hps=tuner.get_best_hyperparameters(num_trials=10)[0]