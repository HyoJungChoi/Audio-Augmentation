def mfm(c1, out_size, name=None):
        t1 = c1[:,:,:,:out_size]
        t2 = c1[:,:,:,out_size:]
        return tf.maximum(t1, t2)
    
    
    

#### LCNN

in1 = Input(shape = input_shape_cqt)
c1 = Conv2D(filters = 32, kernel_size =5, strides=(1, 1), padding='same', activation=None)(in1)
x = mfm(c1, 16, name='mfm1')
x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)


c2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(x)
x = mfm(c2, 16, name='mfm2a')
#x = Dropout(.5)(x)

c2 = Conv2D(filters = 48, kernel_size =3, strides=(1, 1), padding='same', activation=None)(x)
x = mfm(c2, 24, name='mfm2b')
x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
#x = Dropout(.7)(x) 

c2 = Conv2D(filters = 48, kernel_size =1, strides=(1, 1), padding='same', activation=None)(x)
x = mfm(c2, 24, name='mfm3a')
#x = tf.maximum(c2[:,:32],c2[:,32:])

#x = Dropout(.75)(x)
c2 = Conv2D(filters = 64, kernel_size =3, strides=(1, 1), padding='same', activation=None)(x)
x = mfm(c2, 32, name='mfm3b')
x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
x = Dropout(.7)(x) 


c2 = Conv2D(filters = 64, kernel_size =1, strides=(1, 1), padding='same', activation=None)(x)
x = mfm(c2, 32, name='mfm4a')
c2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(x)
x = mfm(c2, 16, name='mfm4b')
x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)
x = Dropout(.7)(x) 


c2 = Conv2D(filters = 32, kernel_size =1, strides=(1, 1), padding='same', activation=None)(x)
x = mfm(c2, 16, name='mfm5a')
c2 = Conv2D(filters = 32, kernel_size =3, strides=(1, 1), padding='same', activation=None)(x)
x = mfm(c2, 16, name='mfm5b')
x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
x = Dropout(.7)(x) 

x = GlobalAveragePooling2D(data_format='channels_last')(x)

f1 = Dense(32*2,activation=None)(x)
x = tf.maximum(f1[:,:32],f1[:,32:])
x = Dropout(.5)(x)
#f2 = linear(x,2,name='fc7')
#x = mfm(f2, 256, name='mfm5')


x = Dense(1, activation = 'sigmoid')(x)

model = keras.models.Model(inputs = in1, outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['AUC'])
model.summary()