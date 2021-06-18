import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
import numpy as np
from cont_data_process import continuous_process
from prepare_data_cube_cont import make_cubes_cont
from prepare_LSTM_data import LSTM_data
from create_model import make_model
# next will be timedist_bi_conv? or cnn_bi_lstm 

#training_filename = 'dataframe_training_new.csv'
training_filename = 'gear_train.csv'
training_df = pd.read_csv(training_filename, sep=',')
attack = training_df[training_df['Attack'] == 'T'].copy()
training_df.drop(attack.index, axis=0, inplace=True)
allRowsTraining = training_df.shape[0]
print(len(training_df[training_df['Attack'] == 'T']))
training_IDs = training_df['ID']
training_IDs = np.array(training_IDs)
#training_dataValues = training_df.drop(["Timestamp", "ID", "DLC", "Attack Window Number", "Normal Window Number", "Data1", "Data2", "Data3", "Data4", "Data5", "Data6", "Data7", "Data8","Total Time", "Attack", "Packet Deltatime"], axis = 1).copy()
#training_dataValues = training_df.drop([ "ID", "Packet Deltatime", "Attack","Attack Window Number", "Normal Window Number","Total Time"], axis = 1).copy()
training_dataValues = training_df.drop([ "Timestamp","ID", "Packet Deltatime", "Attack","Attack Window Number", "Normal Window Number","Total Time","Dataset"], axis = 1).copy()

training_dataValues = training_dataValues.to_numpy() 

n_steps = int(allRowsTraining/10000) # about 10k in each step #127 before
print(n_steps)
training_split = np.array_split(range(allRowsTraining), n_steps)
 
#val_filename = 'dataframe_validation_normal_new.csv'
val_filename = 'gear_val.csv'
val_df = pd.read_csv(val_filename, sep=',')
attack_val = val_df[val_df['Attack'] == 'T'].copy()
val_df.drop(attack_val.index, axis=0, inplace=True)
allRowsVal = val_df.shape[0]
print(len(val_df[val_df['Attack'] == 'T']))
val_IDs = val_df['ID']
val_IDs = np.array(val_IDs)
print(val_IDs)
#val_dataValues = val_df.drop(["Timestamp", "ID", "DLC", "Attack Window Number", "Normal Window Number", "Data1", "Data2", "Data3", "Data4", "Data5", "Data6", "Data7", "Data8","Total Time", "Attack", "Packet Deltatime"], axis = 1).copy()
#val_dataValues = val_df.drop([ "ID", "Packet Deltatime", "Attack","Attack Window Number", "Normal Window Number","Total Time"], axis = 1).copy()
val_dataValues  = val_df.drop([ "Timestamp","ID", "Packet Deltatime", "Attack","Attack Window Number", "Normal Window Number","Total Time","Dataset"], axis = 1).copy()

val_dataValues = val_dataValues.to_numpy()

val_split = np.array_split(range(allRowsVal), n_steps)
 
overlap = 20 
type = 'cnn_lstm'
model = make_model(type) 

es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)
checkpoint_filepath = 'cnnlstm_gear_06-18'   # MISSA INTE ATT ÄNDRA!!!!!!!!!!!
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)


for y in range(n_steps):
    print(f'{y} of {n_steps}')
    training_ind = training_split[y]
    
    print(training_ind)
    val_ind = val_split[y]

    t_IDs = training_IDs[training_ind]
    training_data = training_dataValues[training_ind,:]

    v_IDs = val_IDs[val_ind]
    val_data = val_dataValues[val_ind,:]

    #x_train = LSTM_data(training_data,overlap,training_ind)
    #x_val = LSTM_data(val_data,overlap,val_ind)
    x_train,_ = make_cubes_cont(t_IDs,training_data,40,'cnn_lstm',training_ind,overlap) 
    x_val,_ = make_cubes_cont(v_IDs,val_data,40,'cnn_lstm',val_ind,overlap) 
 
    model.fit(x_train,x_train, validation_data=(x_val, x_val), epochs=100, verbose=2, batch_size = 100, shuffle=False, callbacks = [es,model_checkpoint_callback])



overlap = 20
type = '3dCNN'
model = make_model(type)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)
checkpoint_filepath = '3dcnn_gear_06-18'   # MISSA INTE ATT ÄNDRA!!!!!!!!!!!
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

for y in range(n_steps):
    print(f'{y} of {n_steps}')
    training_ind = training_split[y]
    val_ind = val_split[y]

    t_IDs = training_IDs[training_ind]
    training_data = training_dataValues[training_ind,:]

    v_IDs = val_IDs[val_ind]
    val_data = val_dataValues[val_ind,:]

    #x_train = LSTM_data(training_data,overlap,training_ind)
    #x_val = LSTM_data(val_data,overlap,val_ind)
    x_train,_ = make_cubes_cont(t_IDs,training_data,40,'cnn',training_ind,overlap) 
    x_val,_ = make_cubes_cont(v_IDs,val_data,40,'cnn',val_ind,overlap) 
 
    model.fit(x_train,x_train, validation_data=(x_val, x_val), epochs=100, verbose=2, batch_size = 100, shuffle=False, callbacks = [es,model_checkpoint_callback])

overlap = 20
type = 'timedist_cnn_lstm'
model = make_model(type)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)
checkpoint_filepath = 'timedist_cnn_lstm_gear-06-18'   # MISSA INTE ATT ÄNDRA!!!!!!!!!!!
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

for y in range(n_steps):
    print(f'{y} of {n_steps}')
    training_ind = training_split[y]
    val_ind = val_split[y].copy()

    t_IDs = training_IDs[training_ind].copy()
    training_data = training_dataValues[training_ind,:].copy()

    v_IDs = val_IDs[val_ind].copy()
    val_data = val_dataValues[val_ind,:].copy()

    #x_train = LSTM_data(training_data,overlap)
    #x_val = LSTM_data(val_data,overlap)
    x_train,_ = make_cubes_cont(t_IDs,training_data,40,'timeDist_cnn',training_ind,overlap) 
    x_val,_ = make_cubes_cont(v_IDs,val_data,40,'timeDist_cnn',val_ind,overlap) 
 
    model.fit(x_train,x_train, validation_data=(x_val, x_val), epochs=100, verbose=2, batch_size = 100, shuffle=False, callbacks = [es,model_checkpoint_callback])


overlap = 20
type = 'conv-lstm'
model = make_model(type)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)
checkpoint_filepath = 'conv_lstm_gear-06-18'   # MISSA INTE ATT ÄNDRA!!!!!!!!!!!
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

for y in range(n_steps):
    print(f'{y} of {n_steps}')
    training_ind = training_split[y]
    val_ind = val_split[y].copy()

    t_IDs = training_IDs[training_ind].copy()
    training_data = training_dataValues[training_ind,:].copy()

    v_IDs = val_IDs[val_ind].copy()
    val_data = val_dataValues[val_ind,:].copy()

    #x_train = LSTM_data(training_data,overlap)
    #x_val = LSTM_data(val_data,overlap)
    x_train,_ = make_cubes_cont(t_IDs,training_data,40,'cnn',training_ind,overlap) 
    x_val,_ = make_cubes_cont(v_IDs,val_data,40,'cnn',val_ind,overlap) 
 
    model.fit(x_train,x_train, validation_data=(x_val, x_val), epochs=100, verbose=2, batch_size = 100, shuffle=False, callbacks = [es,model_checkpoint_callback])


training_filename = 'dataframe_training_new.csv'

training_df = pd.read_csv(training_filename, sep=',')
allRowsTraining = training_df.shape[0]

training_IDs = training_df['ID']
training_IDs = np.array(training_IDs)
training_dataValues = training_df.drop([ "ID", "Packet Deltatime", "Attack","Attack Window Number", "Normal Window Number","Total Time"], axis = 1).copy()

training_dataValues = training_dataValues.to_numpy() 

n_steps = int(allRowsTraining/10000) # about 10k in each step #127 before
print(n_steps)
training_split = np.array_split(range(allRowsTraining), n_steps)
 
val_filename = 'dataframe_validation_normal_new.csv'
val_df = pd.read_csv(val_filename, sep=',')
allRowsVal = val_df.shape[0]

val_IDs = val_df['ID']
val_IDs = np.array(val_IDs)
val_dataValues = val_df.drop([ "ID", "Packet Deltatime", "Attack","Attack Window Number", "Normal Window Number","Total Time"], axis = 1).copy()

val_dataValues = val_dataValues.to_numpy()

val_split = np.array_split(range(allRowsVal), n_steps)

overlap = 20
type = 'bi_convLSTM'
model = make_model(type)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)
checkpoint_filepath = 'bi_convLSTM_new_06-18'   # MISSA INTE ATT ÄNDRA!!!!!!!!!!!
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

for y in range(n_steps):
    print(f'{y} of {n_steps}')
    training_ind = training_split[y]
    val_ind = val_split[y]

    t_IDs = training_IDs[training_ind]
    training_data = training_dataValues[training_ind,:]

    v_IDs = val_IDs[val_ind]
    val_data = val_dataValues[val_ind,:]

    #x_train = LSTM_data(training_data,overlap)
    #x_val = LSTM_data(val_data,overlap)
    x_train,_ = make_cubes_cont(t_IDs,training_data,40,'cnn',training_ind,overlap) 
    x_val,_ = make_cubes_cont(v_IDs,val_data,40,'cnn',val_ind,overlap) 
 
    model.fit(x_train,x_train, validation_data=(x_val, x_val), epochs=100, verbose=2, batch_size = 100, shuffle=False, callbacks = [es,model_checkpoint_callback])
