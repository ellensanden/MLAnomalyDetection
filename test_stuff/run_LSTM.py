import numpy as np
import pandas as pd

#mean normal: 0.0209, mean attack: 0.0234 get same results with no overlap
colnames = ["time", "ID", "DLC", "Data1", \
        "Data2", "Data3", "Data4", "Data5", "Data6", "Data7", "Data8", "Attack"]

#nRows = 2000000 #number of rows that you want
df = pd.read_csv('gear_dataset.csv', sep=',', names=colnames)

uniqueIDs = df['ID'].unique() #26 for the entire dataset

#Drop attack packets
attack = df[df['Attack'] == 'T'].copy()
#df.drop(attack.index, axis=0, inplace=True)

#Drop DLC = 2 packets
dlc2 = df[df['DLC'] == 2]
df.drop(dlc2.index, axis=0, inplace=True) #drop all dlc2 indexes

#Pick an ID
id_data= df[df['ID'] == '043f'].copy()
#id_data = df

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
import pickle

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
X_train_samples = overlapping_window(time_steps,20,a) # 40 is no overlap
X_train = storage[X_train_samples,:]
X_train = np.squeeze(X_train)
print(X_train.shape)
X_reversed = np.flip(X_train,1)

n_samples = X_train.shape[0]
n_features = X_train.shape[2]
attack_ind = attack.index

contains_attack = [np.any(np.in1d(x, attack_ind)) for x in X_train_samples]
attack_samples = contains_attack
print(f'number of attack samples {sum(attack_samples)}')
no_attacks = ~np.array(contains_attack)

# run net and save parameters 
modelname = 'LSTM_autoencoder_all_rows_ID_043f'
#modelname = 'LSTM_AE_128_one_ID_june'
CNN =  keras.models.load_model(modelname)

attack_cubes = X_train[attack_samples,:,:]
x_test = X_train[no_attacks,:,:]



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
 

normal_errors = normal_errors.reshape(normal_errors.shape[0],normal_errors.shape[1]*normal_errors.shape[2])


normal_errors = np.abs(normal_errors)
normal_errors = np.sum(normal_errors,axis=1) 
print(normal_errors.shape)

true_attack_errors = attack_cubes-yHat_true_attack



true_attack_errors = true_attack_errors.reshape(true_attack_errors.shape[0],true_attack_errors.shape[1]*true_attack_errors.shape[2])

true_attack_errors = np.abs(true_attack_errors)
true_attack_errors = np.sum(true_attack_errors,axis=1)
print(true_attack_errors.shape)

#maxNormal = np.max(normal_errors) + 0.01*np.max(normal_errors)
#minAttack = np.min(true_attack_errors) - 0.01*np.max(normal_errors)

# save normal errors and attack errors

# make into binary
yHat_true_attack[yHat_true_attack>0.5]=1
yHat_true_attack[yHat_true_attack<0.5]=0
binary_attack_errros = np.sum(attack_cubes-yHat_true_attack)

yHat_normal[yHat_normal>0.5]=1
yHat_normal[yHat_normal<0.5]=0
binary_normal_errros = np.sum(x_test-yHat_normal)

print(f'attack: {binary_attack_errros}, normal: {binary_normal_errros}')

f = open('normal_errors.pckl', 'wb')
pickle.dump(normal_errors, f)
f.close()


f = open('attack_errors.pckl', 'wb')
pickle.dump(true_attack_errors, f)
f.close()