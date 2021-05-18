def process(datafile,rows,no_attack_packets):

    import numpy as np
    import pandas as pd

    colnames = ["time", "ID", "DLC", "Data1", \
            "Data2", "Data3", "Data4", "Data5", "Data6", "Data7", "Data8", "Attack"]

    
    #
    if rows == 'slice':
        df = pd.read_csv(datafile,nrows= 550000, sep=',', names=colnames) #these are for slicing unseen data
        df = df.iloc[50000:100000,:]                                     #these are for slicing unseen data
    else: 
        nRows = rows
        df = pd.read_csv(datafile, nrows = nRows, sep=',', names=colnames)
    
    
    uniqueIDs = df['ID'].unique() #26 for the entire dataset

    #Drop attack packets
    attack = df[df['Attack'] == 'T'].copy()
    if no_attack_packets == True:
        df.drop(attack.index, axis=0, inplace=True)
        print(f'dropped {len(attack)} attack packets')
    else:
        print(f'number of attack packets in data set = {len(attack)}')

    #Drop DLC = 2 packets
    dlc2 = df[df['DLC'] == 2]
    df.drop(dlc2.index, axis=0, inplace=True) #drop all dlc2 indexes


    #Pick an ID
    #id_data= df[df['ID'] == '0140'].copy()
    id_data = df # to use all ids
    ID_vector = df['ID']
    #Just use data values without time, Attack, ID and DLC right now

    dataValues = id_data.drop(["time", "Attack", "ID", "DLC"], axis = 1).copy()

    dataValues = dataValues.to_numpy()


    storage = np.zeros((len(dataValues),64), dtype=int)
    for currentRow in np.arange(len(storage)):
        
        tempString = "".join(dataValues[currentRow])
        formatted = format(int(tempString, base=16), "064b")
        storage[currentRow,:] = np.array(list(formatted), dtype=int)
        
    
    return storage, ID_vector,id_data