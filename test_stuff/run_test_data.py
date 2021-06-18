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

#filename = 'dataframe_test_attack.csv'
filename = 'gear_test.csv'
df = pd.read_csv(filename, sep=',')
allRows = df.shape[0]

IDs = df['ID']
IDs = np.array(IDs)
#IDs = IDs[:148400]
attack = df[df['Attack'] == 'T'].copy()
#attack = attack[:148400]
attack_ind = attack.index

#dataValues = df.drop([ "ID", "Packet Deltatime", "Attack","Attack Window Number", "Normal Window Number"], axis = 1).copy()
dataValues = df.drop([ "Timestamp","ID", "Packet Deltatime", "Attack","Attack Window Number", "Normal Window Number","Total Time","Dataset"], axis = 1).copy()

dataValues = dataValues.to_numpy() 
print(dataValues.shape)
#dataValues = dataValues[:148400]
n_steps = int(allRows/10000) # about 10k in each step #127 before
print(n_steps)
split = np.array_split(range(allRows), n_steps)

type = 'cnn'
#type = 'lstm'
#modelname = '3dCNN_final'
#modelname = 'small_LSTM_final'
#type = 'cnn_lstm'
modelname = 'newnew3dcnn'

model =  keras.models.load_model(modelname)

all_attack_samples = []
errors = []

for y in range(n_steps):
    print(f'{y} of {n_steps}')
    data_ind = split[y]

    t_IDs = IDs[data_ind]
    training_data = dataValues[data_ind,:]
    x_val,samples = make_cubes_cont(t_IDs,training_data,40,type,data_ind) 
    
    contains_attack = [np.any(np.in1d(x, attack_ind)) for x in samples]
    attack_samples = contains_attack
    
   
    yHat_normal = model.predict(x_val) # only normal packets
    all_errors = x_val-yHat_normal

    if type == 'timeDist_cnn' or type == 'cnn':
        all_errors = all_errors.reshape(all_errors.shape[0],all_errors.shape[1]*all_errors.shape[2]*all_errors.shape[3]*all_errors.shape[4])

    if type == 'cnn_lstm':
        all_errors = all_errors.reshape(all_errors.shape[0],all_errors.shape[1]*all_errors.shape[2]*all_errors.shape[3])
    
    if type == 'lstm':
        all_errors = all_errors.reshape(all_errors.shape[0],all_errors.shape[1]*all_errors.shape[2])

    all_errors = np.abs(all_errors)
    all_errors = np.sum(all_errors,axis=1)
    errors.append(all_errors)
    
    all_attack_samples.append(attack_samples)
 
 
f = open('test_errors_newnew3dcnn.pckl', 'wb')
pickle.dump(errors, f)
f.close()


f = open('all_attack_samples_newnew3dcnn.pckl', 'wb')
pickle.dump(all_attack_samples, f)
f.close()

# import pickle
# import keras
# import tensorflow as tf

# from keras.callbacks import EarlyStopping
# import numpy as np
# import pandas as pd 

# n_timesteps = 40

# filename = 'dataframe_test_attack.csv'
# df = pd.read_csv(filename, sep=',')
# allRows = df.shape[0]

# IDs = df['ID']
# attack = df[df['Attack'] == 'Yes'].copy()
# attack_ind = attack.index

# dataValues = df.drop([ "ID", "Packet Deltatime", "Attack","Attack Window Number", "Normal Window Number"], axis = 1).copy()

# dataValues = dataValues.to_numpy() 

# from prepare_data_cube import make_cubes

# #type = 'timeDist_cnn'
# type = 'cnn_lstm'

# x_test,x_train, samples = make_cubes(IDs,dataValues,n_timesteps,type)


# contains_attack = [np.any(np.in1d(x, attack_ind)) for x in samples]

# normal_errors = x_test-yHat_normal

# if type == 'timeDist_cnn' or type == 'cnn':
#     normal_errors = normal_errors.reshape(normal_errors.shape[0],normal_errors.shape[1]*normal_errors.shape[2]*normal_errors.shape[3]*normal_errors.shape[4])

# if type == 'cnn_lstm':
#     normal_errors = normal_errors.reshape(normal_errors.shape[0],normal_errors.shape[1]*normal_errors.shape[2]*normal_errors.shape[3])

# normal_errors = np.abs(normal_errors)
#normal_errors = np.sum(normal_errors,axis=1)
