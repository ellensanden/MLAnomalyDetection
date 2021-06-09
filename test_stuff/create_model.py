def make_model(type):
    from keras.layers import ConvLSTM2D
    from keras.layers import Input
    from keras.layers import Conv3DTranspose
    from keras.layers import Dense
    from keras.models import Model
    from keras.layers import Bidirectional
    from keras.layers import TimeDistributed
    import tensorflow as tf
    from keras.layers import LSTM
    from keras.layers import Dropout
    from keras.layers import Conv3D
    from keras.layers import MaxPool3D
    from tensorflow.python.keras.layers.normalization import BatchNormalization
    from keras.layers import Activation
    print(type)

    if type == 'bi_convLSTM':
    
        input = Input(shape=(40,64,17,1))

        x = Bidirectional(ConvLSTM2D(filters = 2, kernel_size = (11, 9),strides=(2, 2),return_sequences=True,return_state=False),merge_mode='sum')(input)

        x, state_h1, state_c1, state_h2, state_c2 = Bidirectional(ConvLSTM2D(filters = 2, kernel_size = (1, 1),strides=(1, 1),return_sequences=True,return_state=True), merge_mode='sum')(x)

        x = Bidirectional(ConvLSTM2D(filters = 2, kernel_size = (1, 1),strides=(1, 1),return_sequences=True),  merge_mode='sum')(x, initial_state = [state_h1,state_c1, state_h2, state_c2])

        x = Bidirectional(ConvLSTM2D(filters = 2, kernel_size = (9, 3),strides=(2, 1),return_sequences=True,return_state=False),merge_mode='sum')(x)

        x, state_h1, state_c1, state_h2, state_c2 = Bidirectional(ConvLSTM2D(filters = 2, kernel_size = (1, 1),strides=(1, 1),return_sequences=True,return_state=True),merge_mode='sum')(x)

        x = Bidirectional(ConvLSTM2D(filters = 2, kernel_size = (1, 1),strides=(1, 1),return_sequences=True),merge_mode='sum' )(x, initial_state = [state_h1, state_c1, state_h2, state_c2])

        x = Bidirectional(ConvLSTM2D(filters = 2, kernel_size = (3, 1),strides=(1, 1),return_sequences=True),merge_mode='sum')(x)

        x, state_h1, state_c1, state_h2, state_c2 = Bidirectional(ConvLSTM2D(filters = 2, kernel_size = (1, 1),strides=(1, 1),return_sequences=True,return_state=True), merge_mode = 'sum')(x)

        x = Bidirectional(ConvLSTM2D(filters = 2, kernel_size = (1, 1),strides=(1, 1),return_sequences=True), merge_mode='sum')(x, initial_state = [state_h1, state_c1, state_h2, state_c2])


        k1 = 40 - x.shape[1] + 1 
        k2 = 64 - x.shape[2] + 1
        k3 = 17 - x.shape[3] + 1 

        x = Conv3DTranspose(1,kernel_size=(k1,k2,k3), strides=(1, 1, 1))(x) # this should work always as long as strides are 1,1,1
        x = TimeDistributed(Dense(1, activation='sigmoid'))(x)

        CNN = Model(inputs=input, outputs=x,name="CNN")
        CNN.compile(optimizer='adam', loss='BinaryCrossentropy')
        CNN.summary()

        model = CNN
        return model
    #return model
    elif type == 'cannolo_LSTM':
        lstm_initializer = tf.keras.initializers.RandomUniform(minval=-0.5, maxval=0.5)

        # define Encoder
        Inputs = Input(shape=(40,64))
        dense1 =Dense(256, activation='tanh')(Inputs)
        dropout = Dropout(0.2)(dense1)
        lstm1 = LSTM(128,return_sequences=True,kernel_initializer =lstm_initializer, recurrent_initializer=lstm_initializer)(dropout)
        lstm2, state_h, state_c = LSTM(128,return_sequences=True,return_state=True,kernel_initializer =lstm_initializer, recurrent_initializer=lstm_initializer)(lstm1)
        encoder_states = [state_h, state_c]

        # define Decoder
        lstm3 = LSTM(128,return_sequences=True,kernel_initializer =lstm_initializer, recurrent_initializer=lstm_initializer)(lstm2,initial_state=encoder_states)
        lstm4 = LSTM(128,return_sequences=True,kernel_initializer =lstm_initializer, recurrent_initializer=lstm_initializer)(lstm3)
        dense2 = Dense(256, activation='sigmoid')(lstm4)
        output = Dense(64,activation= 'sigmoid')(dense2)

        EncoderDecoder = Model(inputs=Inputs, outputs=output,name="EncoderDecoder")
        EncoderDecoder.compile(optimizer='adam', loss='BinaryCrossentropy')
        EncoderDecoder.summary()

        model = EncoderDecoder

        #return model

    elif type == '3dCNN':
        input = Input(shape=(40,64,17,1))

        x = Conv3D(filters = 5, kernel_size = (21, 41, 9), activation='relu', strides=(1, 1, 1), padding='same')(input) 
        x = MaxPool3D((2,2,2),padding='same')(x)

        x = Conv3D(filters = 5, kernel_size = (11, 21, 3),  activation='relu', strides=(1, 1, 1), padding='same')(x) 
        x = MaxPool3D((2,2,2),padding='same')(x)

        x = Conv3D(filters = 5, kernel_size = (3, 3, 3), strides=(1, 1, 1), activation='relu', padding='same')(x) 
        x = MaxPool3D((1,1,1),padding='same')(x)

        x = Conv3D(filters = 5, kernel_size = (9, 9, 3), strides=(1, 1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPool3D((1,1,1),padding='same')(x)


        x = Conv3DTranspose(5,kernel_size=(9,9,3), strides=(1, 1, 1),activation='relu')(x)

        x = Conv3DTranspose(5,kernel_size=(11,11,11), strides=(1, 1, 1),activation='relu')(x)

        k1 = 40 - x.shape[1] + 1
        k2 = 64 - x.shape[2] + 1
        k3 = 17 - x.shape[3] + 1 

        x = Conv3DTranspose(5,kernel_size=(k1,k2,k3), strides=(1, 1, 1),activation='relu')(x) 
        x = Dense(1,activation = 'sigmoid')(x)
        CNN = Model(inputs=input, outputs=x,name="CNN")
        CNN.compile(optimizer='adam', loss='BinaryCrossentropy')
        CNN.summary()
        model = CNN

    else:
        model = []
        print('wrong type name') 

    return model
