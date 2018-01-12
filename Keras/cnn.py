from keras import Sequential
from keras.layers import Dense,Activation,Convolution2D,MaxPooling2D,Flatten
from matplotlib import pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.utils import np_utils

np.random.seed(1337)
if __name__ == '__main__':
    (x_train,y_train),(x_test,y_test) = mnist.load_data()
    x_train = x_train.reshape(-1,1,28,28)
    x_test = x_test.reshape(-1,1,28,28)
    y_train = np_utils.to_categorical(y_train,num_classes=10)
    y_test = np_utils.to_categorical(y_test,num_classes=10)

    model = Sequential()

    #output (32,28,28)
    model.add(Convolution2D(
        filters=32,
        kernel_size=(5,5),
        padding='same',
        input_shape=(1,28,28)
    ))
    model.add(Activation('relu'))

    #pooling output (32,14,14)
    model.add(MaxPooling2D(
        pool_size=(2,2),
        strides=(2,2),
        padding="same"
    ))

    #conv layer2 output(64,14,14)
    model.add(
        Convolution2D(64,(5,5),padding="same")
    )
    model.add(Activation("relu"))

    #pooling2 output(64,7,7)
    model.add(MaxPooling2D(pool_size=(2,2),padding="same"))

    #full connected1 output(64,7,7)
    model.add(Flatten())
    model.add(Dense(1024))
    model.add((Activation('relu')))

    #full connected2
    model.add(Dense(10))
    model.add(Activation('softmax'))

    #define optimizer
    adam = Adam(lr=1e-4)

    model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
    print('training===================')
    model.fit(x_train,y_train,batch_size=32,nb_epoch=2)

    print('\ntesting===================')
    loss,accuracy = model.evaluate(x_test,y_test)

    print('\nloss=',loss)

    print('\naccuracy=',accuracy)

