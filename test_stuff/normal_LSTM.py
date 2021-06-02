import numpy as np
import pandas as pd
# 20, no first dense, 2000 rows, 10 batch: 0.3845
# 20, with first dense, 2000 rows, 10 batch: 0.1494

# 40, no first dense, 2000 rows, 10 batch:
# 40, with first dense, 2000 rows, 10 batch: 0.1072 (3000 rows: 0.2090)
colnames = ["time", "ID", "DLC", "Data1", \
        "Data2", "Data3", "Data4", "Data5", "Data6", "Data7", "Data8", "Attack"]

#nRows = 1000000 #number of rows that you want
df = pd.read_csv('gear_dataset.csv',nrows = 2000000, sep=',', names=colnames)

uniqueIDs = df['ID'].unique() #26 for the entire dataset

#Drop attack packets
attack = df[df['Attack'] == 'T'].copy()
df.drop(attack.index, axis=0, inplace=True)

#Drop DLC = 2 packets
dlc2 = df[df['DLC'] == 2]
df.drop(dlc2.index, axis=0, inplace=True) #drop all dlc2 indexes

#Pick an ID
#id_data= df[df['ID'] == '0140'].copy()
id_data = df

#Just use data values without time, Attack, ID and DLC right now
dataValues = id_data.drop(["time", "Attack", "ID", "DLC"], axis = 1).copy()
dataValues = dataValues.to_numpy()


storage = np.zeros((len(dataValues),64), dtype=int)
for currentRow in np.arange(len(storage)):
    
    tempString = "".join(dataValues[currentRow])
    formatted = format(int(tempString, base=16), "064b")
    storage[currentRow,:] = np.array(list(formatted), dtype=int)


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
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
import numpy as np
from numpy import array


def overlapping_window (window_size,overlap,seq): # overlap 1 is max. larger number would be less overlap
 
    seq = array([seq[i : i + window_size] for i in range(0, len(seq), overlap)]) 
   
    correct = [len(x)==window_size for x in seq]
    seq = seq[correct]
    seq = np.stack(seq, axis=0 )
    seq = seq.reshape(-1,window_size,1)

    return seq

time_steps = 40
n_rows = storage.shape[0]
n_features = storage.shape[1]
a = np.r_[0:n_rows]
X_train_samples = overlapping_window(time_steps,20,a)
X_train = storage[X_train_samples,:]
X_train = np.squeeze(X_train)
print(X_train.shape)
X_reversed = np.flip(X_train,1)

# best value so far on lstm tuning: 
# uniformMin = -0.89
# uniformMax = 0.07
# lstmUnits = 491
# denseUnits = 2060
# dropout = 0.3
# learning_rate = 0.001
# tuner/epochs = 2
# tuner/initial_e... = 0
# tuner/bracket = 4
# tuner/round = 0

n_samples = X_train.shape[0]
n_features = X_train.shape[2]
lstm_initializer = tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5)
#opt = keras.optimizers.Adam(learning_rate=0.001)

# define Encoder

EncoderInputs = Input(shape=(time_steps,n_features))
dense1 =Dense(256, activation='tanh')(EncoderInputs)
dropout = Dropout(0.2)(dense1)
lstm1 = LSTM(40,return_sequences=True,kernel_initializer =lstm_initializer, recurrent_initializer=lstm_initializer)(dropout)
lstm2, state_h, state_c = LSTM(40,return_sequences=True,return_state=True,kernel_initializer =lstm_initializer, recurrent_initializer=lstm_initializer)(lstm1)
encoder_states = [state_h, state_c]

# define Decoder
  
lstm3,state_h, state_c =  LSTM(40,return_sequences=True,return_state=True,kernel_initializer =lstm_initializer, recurrent_initializer=lstm_initializer)(lstm2,initial_state=encoder_states)
lstm4 = LSTM(40,return_sequences=True,kernel_initializer =lstm_initializer, recurrent_initializer=lstm_initializer)(lstm3,initial_state = [state_h,state_c])
dense2 = Dense(256, activation='sigmoid')(lstm4)
output = Dense(n_features,activation= 'sigmoid')(dense2)

EncoderDecoder = Model(inputs=EncoderInputs, outputs=output,name="EncoderDecoder")
EncoderDecoder.compile(optimizer='adam', loss='BinaryCrossentropy')
EncoderDecoder.summary()

import time

model = EncoderDecoder
es= EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=10)
train_size = int(np.floor(0.7*n_samples))

s=time.time()
checkpoint_filepath = 'LSTM_AE_20'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

model.fit(X_train[0:train_size,:,:], X_train[0:train_size,:,:], validation_data=(X_train[train_size:,:,:], X_train[train_size:,:,:]), epochs=5000, verbose=2, batch_size = 10, shuffle=False, callbacks = [es,model_checkpoint_callback])
model.save('LSTM_AE_40')
e = time.time()
print(f'training time = {e-s} seconds')

