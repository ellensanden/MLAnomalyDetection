def continuous_process(dfSlice):

    import numpy as np
    import pandas as pd
    df = dfSlice

    #Drop attack packets
    attack = df[df['Attack'] == 'T'].copy()
    
    df.drop(attack.index, axis=0, inplace=True)
    print(f'dropped {len(attack)} attack packets')

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


    ID_vector = df['ID']

    dataValues = df.drop(["time", "Attack", "ID", "DLC"], axis = 1).copy()

    dataValues = dataValues.to_numpy()


    storage = np.zeros((len(dataValues),64), dtype=int)
    for currentRow in np.arange(len(storage)):
        
        tempString = "".join(dataValues[currentRow])
        formatted = format(int(tempString, base=16), "064b")
        storage[currentRow,:] = np.array(list(formatted), dtype=int)
        

    return storage, ID_vector,df