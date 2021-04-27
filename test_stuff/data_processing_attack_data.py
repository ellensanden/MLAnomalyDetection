def attack_data(datafile,rows)

    import numpy as np
    import pandas as pd

    colnames = ["time", "ID", "DLC", "Data1", \
            "Data2", "Data3", "Data4", "Data5", "Data6", "Data7", "Data8", "Attack"]
        
    nRows = rows #number of rows that you want
    df = pd.read_csv(datafile, nrows=nRows, sep=',', names=colnames)


    #We want to now get all the attack frames
    all_attack = df[df['Attack'] == 'T'].copy()
    unique_attack_IDs = all_attack['ID'].unique() 
    #Only '043f' that is an attack frame in gear_dataset.csv

    #Drop DLC = 2 packets
    dlc2 = df[df['DLC'] == 2]
    df.drop(dlc2.index, axis=0, inplace=True) #drop all dlc2 indexes

    uniqueIDs = df['ID'].unique() #26 for the entire dataset, 25 without the DLC2

    #Pick an ID
    id_data = df[df['ID'] == '043f'].copy()
    print(len(id_data))
    print(id_data.shape)
    #Split the data from specific ID into normal and attack data
    #T is injected, R is normal
    normal_data = id_data[id_data['Attack'] == 'R'].copy()
    total_number_of_normal_samples = normal_data.shape[0]


    #Just use data values without time, Attack, ID and DLC right now
    hexadecimal_data_normal = normal_data.drop(["time", "Attack", "ID", "DLC"], axis = 1).copy()
    hexadecimal_data_normal = hexadecimal_data_normal.to_numpy()

    #Convert from hex to binary, store in numpy array
    binary_normal_data = np.zeros((total_number_of_normal_samples,64), dtype=int)

    for currentRow in np.arange(total_number_of_normal_samples):
        
        tempString = "".join(hexadecimal_data_normal[currentRow])
        formatted = format(int(tempString, base=16), "064b")
        binary_normal_data[currentRow,:] = np.array(list(formatted), dtype=int)
        

    # Store binary attack data
    attack_data = id_data[id_data['Attack'] == 'T'].copy()
    total_number_of_attack_samples = attack_data.shape[0]


    hexadecimal_data_attack = attack_data.drop(["time", "Attack", "ID", "DLC"], axis = 1).copy()
    hexadecimal_data_attack = hexadecimal_data_attack.to_numpy()

    #Convert from hex to binary, store in numpy array
    binary_attack_data = np.zeros((total_number_of_attack_samples,64), dtype=int)

    for currentRow in np.arange(total_number_of_attack_samples):
        
        tempString = "".join(hexadecimal_data_attack[currentRow])
        formatted = format(int(tempString, base=16), "064b")
        binary_attack_data[currentRow,:] = np.array(list(formatted), dtype=int)

    return binary_attack_data, binary_normal_data
    