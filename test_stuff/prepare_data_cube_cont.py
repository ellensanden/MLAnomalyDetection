def make_cubes_cont(IDs,data,n_timesteps,type):
    import numpy as np
    from numpy import array

    def convert_to_binaryIDs(IDs):

        binaryIDs = np.zeros((len(IDs),16),dtype = int)

        for x in range(len(IDs)): 
            currentHex = IDs[x]
            currentBin = format(int(currentHex, base=16), "016b") 
            binaryIDs[x,:] = np.array(list(currentBin), dtype=int)

        return binaryIDs
    
    # only normal data
        
    def overlapping_window (window_size,overlap,seq): # overlap 1 is max. larger number would be less overlap
    
        seq = array([seq[i : i + window_size] for i in range(0, len(seq), overlap)]) 
    
        correct = [len(x)==window_size for x in seq]
        seq = seq[correct]
        samples = np.stack(seq, axis=0 )

        seq  = np.concatenate((seq), axis=None)

        return seq,samples

   

    n_rows = data.shape[0]
    b = np.r_[0:n_rows]
    overlap = 5
    n_samples,samples = overlapping_window(n_timesteps,overlap,b)

    data = data[n_samples,:]
    data = np.squeeze(data)

    IDs = IDs.reset_index(drop=True) 
    id = convert_to_binaryIDs(IDs)
    id = id[n_samples,:]

    ID_matrix =  array([[id],]*data.shape[1])
    ID_matrix = np.squeeze(ID_matrix)
    ID_matrix = ID_matrix.transpose(1,0,2)

    dataCube = np.zeros((data.shape[0],data.shape[1],id.shape[1]+1))
    dataCube[:,:,1:] = ID_matrix
    dataCube[:,:,0] = data

    n_normal_samples = int(np.floor(dataCube.shape[0]/n_timesteps))
    train_size = int(np.floor(0.7*n_normal_samples))
    last_timestep = n_normal_samples*n_timesteps

    x = dataCube[0:last_timestep,:,:]


    if type == 'cnn':
        x = x.reshape(n_normal_samples,n_timesteps,dataCube.shape[1],dataCube.shape[2],1) 

        print(f'x shape = {x.shape}')

        return x,samples 
    # for cnn lstm
    if type == 'cnn_lstm':

        x = x.reshape(n_normal_samples,dataCube.shape[1],dataCube.shape[2],n_timesteps)
        
        print(f'x shape = {x.shape}')

        return x,samples
    

    # for time dist cnn 
    if type == 'timeDist_cnn':

        x = x.reshape(n_normal_samples,n_timesteps,dataCube.shape[1],dataCube.shape[2],1)
        
        print(f'x shape = {x.shape}')

        return x,samples
    


    return x, samples
