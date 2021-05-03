def make_cubes(IDs,AttackIDs,data,data_with_attack,n_timesteps,type):
    # do cube with stream*64*ID
    import numpy as np
    def convert_from_hex(hex,output_type): # converts the data in hex from hexadecimal to decimal or binary form
        out = np.zeros((hex.size))
        if output_type == 'dec':
            for x in range(hex.size):
                h_value = hex[x]
                out[x] = int(h_value,16)
        else:
            for x in range(hex.size):
                h_value = hex[x]
                binary[x] = bin(int(h_value, 16))[2:]

        return out


    #data = convert_from_hex(Data,'dec')
    IDs = IDs.reset_index(drop=True)

    id = convert_from_hex(IDs,'dec') 

    ID_matrix =  array([[id],]*data.shape[1]).transpose()
    ID_matrix = np.squeeze(ID_matrix)
    dataCube = np.dstack([data,ID_matrix])

    AttackIDs = AttackIDs.reset_index(drop=True)

    Attackid = convert_from_hex(AttackIDs,'dec') 

    ID_matrixA =  array([[Attackid],]*data_with_attack.shape[1]).transpose()
    ID_matrixA = np.squeeze(ID_matrixA)
    dataCubeA = np.dstack([data_with_attack,ID_matrixA])


    if type == cnn:
        n_samples = int(np.floor(dataCube.shape[0]/n_timesteps))

        last_timestep = n_samples*n_timesteps
        x = dataCube[0:last_timestep,:,:]
        x = x.reshape(n_samples,n_timesteps,64,2,1)

        train_size = int(np.floor(0.7*n_samples))
        x_train = x[0:train_size,:,:,:,:]
        x_test = x[train_size:,:,:,:,:]

        print(x_test.shape, x_train.shape)

        n_samples = int(np.floor(dataCubeA.shape[0]/n_timesteps))

        last_timestep = n_samples*n_timesteps
        xA = dataCubeA[0:last_timestep,:,:]
        xA = xA.reshape(n_samples,n_timesteps,64,2,1)
        return x_test,x_train,xA

    # for cnn lstm
    if type == cnn_lstm:
        n_samples = int(np.floor(dataCube.shape[0]/n_timesteps))

        last_timestep = n_samples*n_timesteps
        x = dataCube[0:last_timestep,:,:]
        x = x.reshape(n_samples,64,2,n_timesteps)

        train_size = int(np.floor(0.7*n_samples))
        x_train = x[0:train_size,:,:,:]
        x_test = x[train_size:,:,:,:]

        print(x_test.shape, x_train.shape)


        n_samples = int(np.floor(dataCubeA.shape[0]/n_timesteps))

        last_timestep = n_samples*n_timesteps
        xA = dataCubeA[0:last_timestep,:,:]
        xA = xA.reshape(n_samples,64,2,n_timesteps)
        return x_test,x_train,xA
    

    # for time dist cnn 
    if type == timeDist_cnn:
        n_samples = int(np.floor(dataCube.shape[0]/n_timesteps))

        last_timestep = n_samples*n_timesteps
        x = dataCube[0:last_timestep,:,:]
        x = x.reshape(n_samples,n_timesteps,64,2)

        train_size = int(np.floor(0.7*n_samples))
        x_train = x[0:train_size,:,:,:]
        x_test = x[train_size:,:,:,:]

        print(x_test.shape, x_train.shape)


        n_samples = int(np.floor(dataCubeA.shape[0]/n_timesteps))

        last_timestep = n_samples*n_timesteps
        xA = dataCubeA[0:last_timestep,:,:]
        xA = xA.reshape(n_samples,n_timesteps,64,2)
        return x_test,x_train,xA


    return x_test,x_train,xA

