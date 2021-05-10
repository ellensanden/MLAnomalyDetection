def make_cubes(IDs,AttackIDs,data,data_with_attack,n_timesteps,type,labeled_data):
    # do cube with stream*64*ID
    import numpy as np
    from numpy import array
    # def convert_from_hex(hex,output_type): # converts the data in hex from hexadecimal to decimal or binary form
    #     out = np.zeros((hex.size))
    #     if output_type == 'dec':
    #         for x in range(hex.size):
    #             h_value = hex[x]
    #             out[x] = int(h_value,16)
    #     else:
    #         for x in range(hex.size):
    #             h_value = hex[x]
    #             binary[x] = bin(int(h_value, 16))[2:]

    #     return out

    def convert_to_binaryIDs(IDs):

        binaryIDs = np.zeros((len(IDs),16),dtype = int)

        for x in range(len(IDs)):
            currentHex = IDs[x]
            currentBin = format(int(currentHex, base=16), "016b") 
            binaryIDs[x,:] = np.array(list(currentBin), dtype=int)

        return binaryIDs

    #id = convert_from_hex(IDs,'bin') 
    IDs = IDs.reset_index(drop=True) 
    id = convert_to_binaryIDs(IDs)
    print(f'id.shape = {id.shape}')
    ID_matrix =  array([[id],]*data.shape[1]).transpose()
    ID_matrix =  array([[id],]*data.shape[1])
    print(f'ID_matrix.shape = {ID_matrix.shape}')
    ID_matrix = np.squeeze(ID_matrix)
    print(f'ID_matrix.shape = {ID_matrix.shape}')
    print(f'data shape: {data.shape}')
    #data = data.transpose()
    cube = np.zeros((id.shape[1]+1,64,id.shape[0]))
    #dataCube = np.dstack([data,ID_matrix])
    cube[0:-1,:,:] = ID_matrix
    cube[-1,:,:,:] = data

    AttackIDs = AttackIDs.reset_index(drop=True)
    Attackid = convert_to_binaryIDs(AttackIDs)
    #Attackid = convert_from_hex(AttackIDs,'dec') 

    #ID_matrixA =  array([[Attackid],]*data_with_attack.shape[1]).transpose()
    ID_matrixA =  array([[Attackid],]*data_with_attack.shape[1])
    ID_matrixA = np.squeeze(ID_matrixA)
    data_with_attack = data_with_attack.transpose()
    dataCubeA = np.dstack([data_with_attack,ID_matrixA])
    print(f'a datacube: {dataCubeA.shape}')

    n_samples = int(np.floor(dataCube.shape[0]/n_timesteps))
    train_size = int(np.floor(0.7*n_samples))
    last_timestep = n_samples*n_timesteps
    x = dataCube[0:last_timestep,:,:]

    n_attack_samples = int(np.floor(dataCubeA.shape[0]/n_timesteps))

    last_attack_timestep = n_attack_samples*n_timesteps
    print(f'last time step = {last_timestep}, last attack time step = {last_attack_timestep}')
    xA = dataCubeA[0:last_attack_timestep,:,:]

    if type == 'cnn':

        x = x.reshape(n_samples,n_timesteps,64,17,1)
        x_train = x[0:train_size,:,:,:,:]
        x_test = x[train_size:,:,:,:,:]

        xA = xA.reshape(n_Asamples,n_timesteps,64,17,1)

        print(f'x_test shape = {x_test.shape}, x_train shape = {x_train.shape}, with attack = {xA.shape}')

        return x_test,x_train,xA,last_attack_timestep

    # for cnn lstm
    if type == 'cnn_lstm':

        x = x.reshape(n_samples,64,17,n_timesteps)
        x_train = x[0:train_size,:,:,:]
        x_test = x[train_size:,:,:,:]

        xA = xA.reshape(n_attack_samples,64,17,n_timesteps)
        
        print(f'x_test shape = {x_test.shape}, x_train shape = {x_train.shape}, with attack = {xA.shape}')

        return x_test,x_train,xA,last_attack_timestep
    

    # for time dist cnn 
    if type == 'timeDist_cnn':

        x = x.reshape(n_samples,n_timesteps,64,17,1)
        x_train = x[0:train_size,:,:,:]
        x_test = x[train_size:,:,:,:]
        
        xA = xA.reshape(n_attack_samples,n_timesteps,64,17,1)

        print(f'x_test shape = {x_test.shape}, x_train shape = {x_train.shape}, with attack = {xA.shape}')

        return x_test,x_train,xA,last_attack_timestep
    


    return x_test,x_train,xA,last_attack_timestep

