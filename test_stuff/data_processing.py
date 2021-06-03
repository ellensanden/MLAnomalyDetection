def process(datafile,rows,no_attack_packets):

    import numpy as np
    import pandas as pd

    colnames = ["time", "ID", "DLC", "Data1", \
            "Data2", "Data3", "Data4", "Data5", "Data6", "Data7", "Data8", "Attack"]

    
    #
    if rows == 'slice': 
        df = pd.read_csv(datafile,nrows= 70000, sep=',', names=colnames) #these are for slicing unseen data
        df = df.iloc[50000:70000,:]                                     #these are for slicing unseen data
    
    elif rows == 'all':
        df = pd.read_csv(datafile, sep=',', names=colnames)

    else: 
        nRows = rows
        df = pd.read_csv(datafile, nrows = nRows, sep=',', names=colnames)
    
    
    #Drop attack packets
    attack = df[df['Attack'] == 'T'].copy()
    if no_attack_packets == True:
        df.drop(attack.index, axis=0, inplace=True)
        print(f'dropped {len(attack)} attack packets')
    else:
        print(f'number of attack packets in data set = {len(attack)}')

    #Drop DLC = 2 packetss
    #dlc2 = df[df['DLC'] == 2]
    #df.drop(dlc2.index, axis=0, inplace=True) #drop all dlc2 indexes
    unique_DLCs = df['DLC'].unique()
    for current_DLC in unique_DLCs:
        
        current_indexes = df[df['DLC'] == current_DLC].index
        
        if(current_DLC != 8):
            #Set Attack correctly
            where_attack_at_string = 'Data' + str(current_DLC+1)
            df.loc[current_indexes,'Attack'] = df.loc[current_indexes, where_attack_at_string]
            
            #Set the Data to 00
            for j in range(current_DLC+1,9):
                write_to_column = 'Data' + str(j)
                df.loc[current_indexes, write_to_column] = '00'


    #Pick an ID
    #id_data= df[df['ID'] == '0140'].copy()
    id_data = df # to use all ids 
    ID_vector = df['ID']

    dataValues = id_data.drop(["time", "Attack", "ID", "DLC"], axis = 1).copy()

    dataValues = dataValues.to_numpy()


    storage = np.zeros((len(dataValues),64), dtype=int)
    for currentRow in np.arange(len(storage)):
        
        tempString = "".join(dataValues[currentRow])
        formatted = format(int(tempString, base=16), "064b")
        storage[currentRow,:] = np.array(list(formatted), dtype=int)
        
    # storage = storage.astype('int16')
    # ID_vector = ID_vector.astype('int16')
    # id_data = id_data.astype('int16')

    return storage, ID_vector,id_data