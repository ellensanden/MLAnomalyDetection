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
    
    # only normal data

    IDs = IDs.reset_index(drop=True) 
    #id = convert_from_hex(IDs,'dec') 
    id = convert_to_binaryIDs(IDs)
    #ID_matrix =  array([[id],]*data.shape[1]).transpose()
    ID_matrix =  array([[id],]*data.shape[1])
    ID_matrix = np.squeeze(ID_matrix)
    ID_matrix = ID_matrix.transpose(1,0,2)

    dataCube = np.zeros((data.shape[0],data.shape[1],id.shape[1]+1))
    #dataCube = np.dstack([data,ID_matrix])
    dataCube[:,:,1:] = ID_matrix
    dataCube[:,:,0] = data

    n_samples = int(np.floor(dataCube.shape[0]/n_timesteps))
    train_size = int(np.floor(0.7*n_samples))
    last_timestep = n_samples*n_timesteps

    x = dataCube[0:last_timestep,:,:]


    # normal and attack data

    AttackIDs = AttackIDs.reset_index(drop=True)
    Attackid = convert_to_binaryIDs(AttackIDs) 
   
    ID_matrixA =  array([[Attackid],]*data_with_attack.shape[1])
    ID_matrixA = np.squeeze(ID_matrixA)
    ID_matrixA = ID_matrixA.transpose(1,0,2)
    
    dataCubeA = np.zeros((data_with_attack.shape[0],data_with_attack.shape[1],Attackid.shape[1]+1))
    dataCubeA[:,:,1:] = ID_matrixA
    dataCubeA[:,:,0] = data_with_attack

    n_attack_samples = int(np.floor(dataCubeA.shape[0]/n_timesteps))
    last_attack_timestep = n_attack_samples*n_timesteps
    
    print(f'last time step = {last_timestep}, last attack time step = {last_attack_timestep}')
    xA = dataCubeA[0:last_attack_timestep,:,:]

    if type == 'cnn':

        x = x.reshape(n_samples,n_timesteps,dataCube.shape[1],dataCube.shape[2],1) # does this give correct results?
        x_train = x[0:train_size,:,:,:,:]
        x_test = x[train_size:,:,:,:,:]

        xA = xA.reshape(n_attack_samples,n_timesteps,dataCubeA.shape[1],dataCubeA.shape[2],1)

        print(f'x_test shape = {x_test.shape}, x_train shape = {x_train.shape}, with attack = {xA.shape}')

        return x_test,x_train,xA,last_attack_timestep

    # for cnn lstm
    if type == 'cnn_lstm':

        x = x.reshape(n_samples,dataCube.shape[1],dataCube.shape[2],n_timesteps)
        x_train = x[0:train_size,:,:,:]
        x_test = x[train_size:,:,:,:]

        xA = xA.reshape(n_attack_samples,dataCubeA.shape[1],dataCubeA.shape[2],n_timesteps)
        
        print(f'x_test shape = {x_test.shape}, x_train shape = {x_train.shape}, with attack = {xA.shape}')

        return x_test,x_train,xA,last_attack_timestep
    

    # for time dist cnn 
    if type == 'timeDist_cnn':

        x = x.reshape(n_samples,n_timesteps,dataCube.shape[1],dataCube.shape[2],1)
        x_train = x[0:train_size,:,:,:]
        x_test = x[train_size:,:,:,:]
        
        xA = xA.reshape(n_attack_samples,n_timesteps,dataCubeA.shape[1],dataCubeA.shape[2],1)

        print(f'x_test shape = {x_test.shape}, x_train shape = {x_train.shape}, with attack = {xA.shape}')

        return x_test,x_train,xA,last_attack_timestep
    


    return x_test,x_train,xA,last_attack_timestep

