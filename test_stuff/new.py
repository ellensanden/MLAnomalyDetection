import pandas as pd
import tensorflow as tf
import keras
import pickle
from keras.callbacks import EarlyStopping
import numpy as np
from cont_data_process import continuous_process
from prepare_data_cube_cont import make_cubes_cont
from prepare_LSTM_data import LSTM_data
from create_model import make_model


filename = 'gear_val.csv'
df = pd.read_csv(filename, sep=',')
allRows = df.shape[0]

IDs = df['ID']
attack = df[df['Attack'] == 'T'].copy()
attack_ind = attack.index

dataValues  = df.drop([ "Timestamp","ID", "Packet Deltatime", "Attack","Attack Window Number", "Normal Window Number","Total Time","Dataset"], axis = 1).copy()

dataValues = dataValues.to_numpy() 
print(dataValues.shape)
n_steps = int(allRows/10000) # about 10k in each step #127 before
print(n_steps)
split = np.array_split(range(allRows), n_steps)

type = 'cnn'
#type = 'timeDist_cnn'
#type = 'lstm'
#modelname = '3dCNN_final'
#modelname = 'small_LSTM_final'
#modelname = 'cannolo_LSTM_final'
modelname = '3dCNN_gear_FINAL'
#modelname = 'conv-lstm_gear_FINAL'
#modelname = 'timedist_2d_cnn_lstm_gear_FINAL'
#modelname = '3dCNN_gear_FINAL'
model =  keras.models.load_model(modelname)

all_attack_errors = []
all_normal_errors = []

for y in range(n_steps):
    print(f'{y} of {n_steps}')
    data_ind = split[y]

    t_IDs = IDs[data_ind]
    training_data = dataValues[data_ind,:]
    # LSTM:
    #x_val,samples = LSTM_data(training_data,40,data_ind)
  
    x_val,samples = make_cubes_cont(t_IDs,training_data,40,type,data_ind) 

    contains_attack = [np.any(np.in1d(x, attack_ind)) for x in samples]
    attack_samples = contains_attack
    no_attacks = ~np.array(contains_attack)
    # attack_cubes = x_val[attack_samples,:,:] 
    # x_normal = x_val[no_attacks,:,:]

    attack_cubes = x_val[attack_samples,:,:,:,:]
    x_normal = x_val[no_attacks,:,:,:,:]

    yHat_normal = model.predict(x_normal) # only normal packets
    normal_errors = x_normal-yHat_normal

    if type == 'timeDist_cnn' or type == 'cnn':
        normal_errors = normal_errors.reshape(normal_errors.shape[0],normal_errors.shape[1]*normal_errors.shape[2]*normal_errors.shape[3]*normal_errors.shape[4])

    if type == 'cnn_lstm':
        normal_errors = normal_errors.reshape(normal_errors.shape[0],normal_errors.shape[1]*normal_errors.shape[2]*normal_errors.shape[3])
    if type == 'lstm':
        normal_errors = normal_errors.reshape(normal_errors.shape[0],normal_errors.shape[1]*normal_errors.shape[2])

    normal_errors = np.abs(normal_errors)
    normal_errors = np.sum(normal_errors,axis=1)
    all_normal_errors.append(normal_errors)
    if attack_cubes.shape[0]!=0:
        yHat_attack = model.predict(attack_cubes) # only attack packets
        attack_errors = attack_cubes-yHat_attack
        if type == 'timeDist_cnn' or type =='cnn':
            attack_errors = attack_errors.reshape(attack_errors.shape[0],attack_errors.shape[1]*attack_errors.shape[2]*attack_errors.shape[3] *attack_errors.shape[4])

        if type == 'cnn_lstm':
            attack_errors = attack_errors.reshape(attack_errors.shape[0],attack_errors.shape[1]*attack_errors.shape[2]*attack_errors.shape[3])
        if type == 'lstm':
            attack_errors = attack_errors.reshape(attack_errors.shape[0],attack_errors.shape[1]*attack_errors.shape[2])

        attack_errors = np.abs(attack_errors)
        attack_errors = np.sum(attack_errors,axis=1)
    #x_train = LSTM_data(training_data,overlap)
    #x_val = LSTM_data(val_data,overlap)
        all_attack_errors.append(attack_errors)
 
 

f = open('3dcnnattack.pckl', 'wb')
pickle.dump(all_normal_errors, f)
f.close()


f = open('3dcnnnormal.pckl', 'wb')
pickle.dump(all_attack_errors, f)
f.close()