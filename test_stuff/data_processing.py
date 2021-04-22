def process(datafile,rows):

    import numpy as np
    import pandas as pd

    colnames = ["time", "ID", "DLC", "Data1", \
            "Data2", "Data3", "Data4", "Data5", "Data6", "Data7", "Data8", "Attack"]

    #nRows = 100000 #number of rows that you want
    nRows = rows
    df = pd.read_csv(datafile, nrows = nRows, sep=',', names=colnames)
    #df = pd.read_csv('gear_dataset.csv', sep=',', names=colnames)

    uniqueIDs = df['ID'].unique() #26 for the entire dataset

    #Drop attack packets
    attack = df[df['Attack'] == 'T'].copy()
    df.drop(attack.index, axis=0, inplace=True)

    #Drop DLC = 2 packets
    dlc2 = df[df['DLC'] == 2]
    df.drop(dlc2.index, axis=0, inplace=True) #drop all dlc2 indexes

    #Reset index from 1 to n (not needed actually, so commenting out)
    #df.set_index(np.arange(len(df)), inplace=True)

    #Pick an ID
    #id_data= df[df['ID'] == '0140'].copy()
    id_data = df # to use all ids
    #Just use data values without time, Attack, ID and DLC right now
    dataValues = id_data.drop(["time", "Attack", "ID", "DLC"], axis = 1).copy()
    #dataValues.to_csv (r'one_id.csv', index=None)

    dataValues = dataValues.to_numpy()


    storage = np.zeros((len(dataValues),64), dtype=int)
    for currentRow in np.arange(len(storage)):
        
        tempString = "".join(dataValues[currentRow])
        formatted = format(int(tempString, base=16), "064b")
        storage[currentRow,:] = np.array(list(formatted), dtype=int)
        

    return storage