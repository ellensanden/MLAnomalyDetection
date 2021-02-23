# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Import and format data
# ## These  (until *) don't need to be run

# %%
import pandas as pd
# names = ['Time','ID','Data']
read_file = pd.read_csv("candump-2021-02-08_150302.log", header = None)
read_file.to_csv (r'can_data.csv', index=None)
can_data = pd.read_csv("can_data.csv")


# %%
can_data

# %% [markdown]
# ### remove vcan0

# %%
text = open("can_data.csv", "r")
text = ''.join([i for i in text])     .replace("vcan0", ",")
x = open("can_data.csv","w")
x.writelines(text)
x.close()

# %% [markdown]
# ### remove hash

# %%
text = open("can_data.csv", "r")
text = ''.join([i for i in text])     .replace("#", ",")
x = open("can_data.csv","w")
x.writelines(text)
x.close()

# %% [markdown]
# ### remove parenthesis

# %%
text = open("can_data.csv", "r")
text = ''.join([i for i in text])     .replace("(", "")
x = open("can_data.csv","w")
x.writelines(text)
x.close()

text = open("can_data.csv", "r")
text = ''.join([i for i in text])     .replace(")", "")
x = open("can_data.csv","w")
x.writelines(text)
x.close()


# %%
can_data = pd.read_csv("can_data.csv",names = ['Time','ID','Data'])
can_data.to_csv (r'can_data.csv', index=None)
print(can_data)

# %% [markdown]
# # *

# %%
import pandas as pd
can_data = pd.read_csv("can_data.csv")
print(can_data)


# %%
Time = can_data['Time']
Time = Time[1:-1]
Time = Time.reset_index(drop=True)

ID = can_data['ID']
ID = ID[1:-1]
ID = ID.reset_index(drop=True)

Data = can_data['Data']
Data = Data[1:-1]
Data = Data.reset_index(drop=True)



# %%
import numpy as np

def delta_time(Time): # calculates the time between two subsequent messages
    delta = np.zeros((Time.size))
    for x in range(Time.size-1):

       delta[x] = Time[x+1]-Time[x]

    return delta   

delta = delta_time(Time)


# %%
def convert_from_hex(hex): # converts the data in hex from hexadecimal to decimal form
    dec = np.zeros((hex.size))

    for x in range(hex.size):
        h_value = hex[x]
        dec[x] = int(h_value,16)
    return dec


data = convert_from_hex(Data)
id = convert_from_hex(ID)

# %% [markdown]
# ## Normalize data

# %%
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))
data = data.reshape(-1, 1)
data = scaler.fit_transform(data)
id = id.reshape(-1, 1)
id = scaler.fit_transform(id)

# %% [markdown]
# ## Visualize data

# %%
from matplotlib import pyplot as plt

plt.figure()
plt.plot(Time[0:10000],data[0:10000],'o')
plt.xlabel('Time')
plt.ylabel('Data')

plt.figure()
plt.plot(Time[0:10000],id[0:10000],'o')
plt.xlabel('Time')
plt.ylabel('ID')

plt.figure()
plt.plot(data[0:10000],id[0:10000],'o')
plt.xlabel('Data')
plt.ylabel('ID')



# %%
steps = range(Time.size)
plt.figure()
plt.plot(steps[0:100000],delta[0:100000])


# %%
def sort_IDs(id): # returns groups of indices of unique ids in list 

    idx_sort = np.argsort(id)
    sorted_ids = id[idx_sort]
    vals, idx_start, count = np.unique(sorted_ids, return_counts=True, return_index=True)
    indices = np.split(idx_sort, idx_start[1:])

    print(f'test gitignore file')
    return indices




# %%
transposed_id = np.transpose(id) # not sure why it needs to be transposed but it doesn't work otherwise
indices = sort_IDs(transposed_id[0][0:30])
print(indices)


# %%
def get_data_stream(indices,data): # collects the unique ids with their respective data streams

    num_unique_ids = len(indices)
    all_ids = []
    for x in range(num_unique_ids):
         
         id_data = data[indices[:][x]]
         all_ids.append(id_data)

    return all_ids
   


# %%
all_id_data = get_data_stream(indices,data)


# %%
#id_data = np.concatenate((id[0:-1],data[0:-1]),axis=1)


# %%
def combine_ids_data(indices,id,all_id_data):
    


# %%
id[indices[:][2]]

# %% [markdown]
# ## Copied

# %%
# split a univariate sequence into samples
from numpy import array


def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)


# %%
n_steps = 5
X, y = split_sequence(id[0:10000], n_steps)


# %%
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense


# %%
# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))


# %%
# define model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


# %%
# fit model

model.fit(X, y, epochs=20, verbose=0)


# %%

# demonstrate prediction
x_input = X[50:55,1,:]
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)


# %%
print( X[56,1,:]-yhat)
print( X[56,1,:])


# %%
train_X = X
train_y = y
test_X = X[50:55,1,:]
test_y = X[56,1,:]
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# %% [markdown]
# 
# %% [markdown]
# ## ^copied

