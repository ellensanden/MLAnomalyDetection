import pandas as pd
import keras
import pickle
import numpy as np
from prepare_data_cube import make_cubes
from prepare_LSTM_data import LSTM_data

# the data set to test on
filename = 'gear_test.csv'

df = pd.read_csv(filename, sep=',')
allRows = df.shape[0]

IDs = df['ID']
IDs = np.array(IDs)

attack = df[df['Attack'] == 'T'].copy() # this is 'Yes' instead of 'T' for some data sets
attack_ind = attack.index

#there will be different column names to drop here depending on data set
dataValues = df.drop([ "ID", "Packet Deltatime", "Attack","Attack Window Number", "Normal Window Number"], axis = 1).copy()

dataValues = dataValues.to_numpy() 
print(dataValues.shape)

n_steps = int(allRows/10000) # about 10k in each step 
print(n_steps)
split = np.array_split(range(allRows), n_steps)

type = 'cnn_lstm' # type of the trained model
modelname = 'trained_model' # the name of the trained model

model =  keras.models.load_model(modelname)

all_attack_samples = []
errors = []
overlap = 40 # overlap = 40 means no overlap 
for y in range(n_steps):
    print(f'{y} of {n_steps}')
    data_ind = split[y]

    t_IDs = IDs[data_ind]
    training_data = dataValues[data_ind,:]

    if type == 'small_lstm' or type == 'large_lstm':
        x_val,samples = LSTM_data(training_data,overlap,data_ind)

    else:  
        x_val,samples = make_cubes(t_IDs,training_data,40,type,data_ind,overlap) 
    
    contains_attack = [np.any(np.in1d(x, attack_ind)) for x in samples]
    attack_samples = contains_attack
    
   
    yHat_normal = model.predict(x_val) # only normal packets
    all_errors = x_val-yHat_normal

    if type == 'large_lstm' or type == 'small_lstm':
        all_errors = all_errors.reshape(all_errors.shape[0],all_errors.shape[1]*all_errors.shape[2])

    elif type == '2d_cnn_lstm':
        all_errors = all_errors.reshape(all_errors.shape[0],all_errors.shape[1]*all_errors.shape[2]*all_errors.shape[3])
    
    else:
        all_errors = all_errors.reshape(all_errors.shape[0],all_errors.shape[1]*all_errors.shape[2]*all_errors.shape[3]*all_errors.shape[4])

    all_errors = np.abs(all_errors)
    all_errors = np.sum(all_errors,axis=1)
    errors.append(all_errors)
    
    all_attack_samples.append(attack_samples)
 
saved_test_errors = 'test_errors.pckl' # name of the saved test errors
f = open(saved_test_errors, 'wb')
pickle.dump(errors, f)
f.close()

saved_true_attack_samples = 'attack_samples.pckl' # name of the saved true attack samples (for later evaluation)
f = open(saved_true_attack_samples, 'wb')
pickle.dump(all_attack_samples, f)
f.close()


