# Definition of Keras ConvNet architecture

from __future__ import print_function

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Conv2D, MaxPooling2D, Reshape, UpSampling2D, Flatten
from keras.models import Model 
from keras import backend as K
from keras import metrics
from keras.datasets import mnist

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy.random as random

batch_size = 100
original_dim = 784
latent_dim = 2
intermediate_dim = 16
epochs = 50 # 50
epsilon_std = 1.0

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., # 2 * 256
                              stddev=epsilon_std)

    return z_mean + K.exp(z_log_var / 2) * epsilon

input_shape = (28, 28, 1)
inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
# shape info needed to build decoder model
shape = K.int_shape(x) 
# generate latent vector Q(z|X)
x = Flatten()(x)
x = Dense(intermediate_dim, activation='relu')(x)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# adding an intermediate input layer
z_inputs = Input(shape=(latent_dim,), name='decoder_input')
x = Dense(intermediate_dim, activation='relu')(z_inputs)
x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(x)
x = Reshape((shape[1], shape[2], shape[3]))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

decoder = Model(z_inputs, x)
outputs = decoder(z)
vae = Model(inputs, outputs, name='vae')

# Compute VAE loss
xent_loss = original_dim * metrics.binary_crossentropy(K.flatten(inputs), K.flatten(outputs))
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
vae.summary()

# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None))

#-------- My Code -------#

def run_encoding():
	encoder = Model(inputs, z)
	return encoder.predict(x_test)

def QuestC():
	test_latent = run_encoding()

	fig, ax = plt.subplots()
	scatter = ax.scatter(test_latent[:, 0], test_latent[:, 1], c=y_test, \
		cmap=cm.get_cmap('rainbow'), edgecolor='black')

	fig.colorbar(scatter)
	ax.grid()
	ax.set_title('(G_c) Test Set in Latent Space')
	fig.savefig("QuestionG_c")
	# plt.show()

	for i in range(10):
		lat_image = test_latent[np.argmax(y_test == i)]
		print("Digit {} Latent Space Coord. {}".format(i, lat_image))

def QuestD():
	z_sample = np.array([[0.5, 0.2]])
	x_decoded = decoder.predict(z_sample)
	fig, ax = plt.subplots()
	ax.imshow(x_decoded.reshape(28, 28))
	ax.set_title('(G_d) Generated Digit')
	fig.savefig("QuestionG_d")
	# plt.show()

def QuestE():
	test_latent = run_encoding()

	x1, y1 = test_latent[np.argmax(y_test == 0)]
	x2, y2 = test_latent[np.argmax(y_test == 1)]
	a = (y1 - y2) / (x1 - x2)
	b = y1 - a * x1

	x_decoded_list = []
	for i in range(10):
		rand_x = np.random.uniform(x1, x2)
		rand_y = a * rand_x + b

		z_sample = np.array([[rand_x, rand_y]])
		x_decoded = decoder.predict(z_sample)
		x_decoded_list.append(x_decoded)
	
	concat_images = x_decoded_list[0].reshape((28,28))

	for i in range(1, 10):
		concat_images = np.concatenate((concat_images, x_decoded_list[i].reshape(28,28)), axis=1)
		plt.imshow(concat_images)

	plt.title('(G_e) Image Sequence')
	plt.savefig("QuestionG_e")
	# plt.show()

if __name__ == '__main__':
	QuestC()
	QuestD()
	QuestE()

