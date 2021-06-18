def LSTM_data(data,overlap,data_ind):
    import numpy as np
    import pandas as pd
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

 
    def overlapping_window (window_size,overlap,seq,original_ind): # overlap 1 is max. larger number would be less overlap
    
        seq = array([seq[i : i + window_size] for i in range(0, len(seq), overlap)]) 
        original_ind = array([original_ind[i : i + window_size] for i in range(0, len(original_ind), overlap)])
        correct = [len(x)==window_size for x in seq]
        original_correct = [len(x)==window_size for x in original_ind]
        original_ind = original_ind[original_correct]
        original_samples = np.stack(original_ind, axis=0 )

        seq = seq[correct]
        seq = np.stack(seq, axis=0 )
        seq = seq.reshape(-1,window_size,1)

        return seq, original_samples

    time_steps = 40
    n_rows = data.shape[0]
    a = np.r_[0:n_rows]
    x_samples,original_samples = overlapping_window(time_steps,overlap,a,data_ind)
    x = data[x_samples,:]
    x = np.squeeze(x)

    return x, original_samples
   
    

