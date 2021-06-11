import pandas as pd
import tensorflow as tf
from keras.callbacks import EarlyStopping
import numpy as np
from cont_data_process import continuous_process
from prepare_data_cube_cont import make_cubes_cont
from prepare_LSTM_data import LSTM_data
from create_model import make_model
# next will be timedist_bi_conv? or cnn_bi_lstm 

training_filename = 'dataframe_training_new.csv'
training_df = pd.read_csv(training_filename, sep=',')
allRowsTraining = training_df.shape[0]

training_IDs = training_df['ID']
#training_dataValues = training_df.drop(["Timestamp", "ID", "DLC", "Attack Window Number", "Normal Window Number", "Data1", "Data2", "Data3", "Data4", "Data5", "Data6", "Data7", "Data8","Total Time", "Attack", "Packet Deltatime"], axis = 1).copy()
training_dataValues = training_df.drop([ "ID", "Packet Deltatime", "Attack","Attack Window Number", "Normal Window Number","Total Time"], axis = 1).copy()

training_dataValues = training_dataValues.to_numpy() 

n_steps = int(allRowsTraining/10000) # about 10k in each step #127 before
print(n_steps)
training_split = np.array_split(range(allRowsTraining), n_steps)
 
val_filename = 'dataframe_validation_normal_new.csv'
val_df = pd.read_csv(val_filename, sep=',')
allRowsVal = val_df.shape[0]

val_IDs = val_df['ID']
#val_dataValues = val_df.drop(["Timestamp", "ID", "DLC", "Attack Window Number", "Normal Window Number", "Data1", "Data2", "Data3", "Data4", "Data5", "Data6", "Data7", "Data8","Total Time", "Attack", "Packet Deltatime"], axis = 1).copy()
val_dataValues = val_df.drop([ "ID", "Packet Deltatime", "Attack","Attack Window Number", "Normal Window Number","Total Time"], axis = 1).copy()

val_dataValues = val_dataValues.to_numpy()

val_split = np.array_split(range(allRowsVal), n_steps)

#model =  keras.models.load_model(modelname)


type1 = 'bi_convLSTM'
model1 = make_model(type1)

type2 = 'timedist_cnn_lstm' 
model2 = make_model(type2) 

es1 = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)
checkpoint_filepath1 = 'bi_convLSTM_final'   # MISSA INTE ATT ÄNDRA!!!!!!!!!!!
model_checkpoint_callback1 = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath1,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

es2 = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)
checkpoint_filepath2 = 'timedist_cnn_lstm_final'   # MISSA INTE ATT ÄNDRA!!!!!!!!!!!
model_checkpoint_callback2 = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath2,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)

for y in range(n_steps):
    print(f'{y} of {n_steps}')
    training_ind = training_split[y]
    val_ind = val_split[y]
    
    #dfSlice = df.iloc[slice1:slice2,:].copy()
    #data, ID_vector,_ = continuous_process(dfSlice)

    t_IDs = training_IDs[training_ind]
    training_data = training_dataValues[training_ind,:]

    v_IDs = val_IDs[val_ind]
    val_data = val_dataValues[val_ind,:]

    #x_train = LSTM_data(training_data,overlap)
    #x_val = LSTM_data(val_data,overlap)
    x_train,_ = make_cubes_cont(t_IDs,training_data,40,'cnn') 
    x_val,_ = make_cubes_cont(v_IDs,val_data,40,'cnn') 
 
    model2.fit(x_train,x_train, validation_data=(x_val, x_val), epochs=100, verbose=2, batch_size = 100, shuffle=False, callbacks = [es2,model_checkpoint_callback2])
    model1.fit(x_train,x_train, validation_data=(x_val, x_val), epochs=100, verbose=2, batch_size = 100, shuffle=False, callbacks = [es1,model_checkpoint_callback1])






   