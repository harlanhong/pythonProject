import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input
import matplotlib.pyplot as plt

# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# X shape (60,000 28x28), y shape (10,000, )
(x_train, _), (x_test, y_test) = mnist.load_data()

# data pre-processing
x_train = x_train.astype('float32') / 255. - 0.5       # minmax_normalized
x_test = x_test.astype('float32') / 255. - 0.5         # minmax_normalized
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))
print(x_train.shape)
print(x_test.shape)


"""
(60000, 784)
(10000, 784)
"""


encoding_dim = 2

input_img = Input(shape=(784,))

#encoder layer

encoded = Dense(128,activation='relu')(input_img)
encoded = Dense(64,activation='relu')(encoded)
encoded = Dense(10,activation='relu')(encoded)
encoded_output = Dense(encoding_dim,)(encoded)

#decoder layers

decoded = Dense(10,activation='relu')(encoded_output)
decoded = Dense(64,activation='relu')(decoded)
decoded = Dense(128,activation='relu')(decoded)
decoded = Dense(784,activation='tanh')(decoded)

#construct model

autoencoder = Model(inputs=input_img,outputs=decoded)

encoder = Model(inputs=input_img,outputs=encoded_output)

autoencoder.compile(optimizer='adam',loss='mse')

autoencoder.fit(x_train,x_train,
                nb_epoch=20,
                batch_size=256,
                shuffle=True)
encoded_imgs = encoder.predict(x_test)
plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test)
plt.colorbar()
plt.show()