'''This script demonstrates how to build a variational autoencoder with Keras.

 #Reference

 - Auto-Encoding Variational Bayes
   https://arxiv.org/abs/1312.6114
'''

from __future__ import print_function

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import metrics
from keras.datasets import mnist

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy.random as random
import sys

batch_size = 100
original_dim = 784
latent_dim = 2
intermediate_dim = 256
epochs = 50
epsilon_std = 1.0

x = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(x) # 256 * 784
z_mean = Dense(latent_dim)(h) # 2 * 256
z_log_var = Dense(latent_dim)(h) # 2 * 256

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., # 2 * 256
                              stddev=epsilon_std)

    return z_mean + K.exp(z_log_var / 2) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_h = Dense(intermediate_dim, activation='relu') # 256
decoder_mean = Dense(original_dim, activation='sigmoid') # 784
h_decoded = decoder_h(z) # 2 * 256
x_decoded_mean = decoder_mean(h_decoded) # 784 * 256

# instantiate VAE model
vae = Model(x, x_decoded_mean)

# Compute VAE loss
xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean)
kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
vae.summary()


# train the VAE on MNIST digits
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

vae.fit(x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, None))

##### my code ######

def run_encoding():
	encoder = Model(x, z)
	return encoder.predict(x_test)

def QuestC(fig_name):
	test_latent = run_encoding()

	fig, ax = plt.subplots()
	scatter = ax.scatter(test_latent[:, 0], test_latent[:, 1], c=y_test, \
		cmap=cm.get_cmap('rainbow'), edgecolor='black')

	fig.colorbar(scatter)
	ax.grid()
	ax.set_title('Test Set in Latent Space')
	fig.savefig(fig_name)
	# plt.show()

	for i in range(10):
		lat_image = test_latent[np.argmax(y_test == i)]
		print("Digit {} Latent Space Coord. {}".format(i, lat_image))

def build_generator():
	decoder_input = Input(shape=(latent_dim,))
	_h_decoded = decoder_h(decoder_input)
	_x_decoded_mean = decoder_mean(_h_decoded)
	return Model(decoder_input, _x_decoded_mean)

def QuestD(fig_name):
	generator = build_generator()
	z_sample = np.array([[0.5, 0.2]])
	x_decoded = generator.predict(z_sample)
	fig, ax = plt.subplots()
	ax.imshow(x_decoded.reshape(28, 28))
	ax.set_title('Generated Digit')
	fig.savefig(fig_name)
	# plt.show()

def QuestE(fig_name):
	test_latent = run_encoding()

	x1, y1 = test_latent[np.argmax(y_test == 0)]
	x2, y2 = test_latent[np.argmax(y_test == 1)]
	a = (y1 - y2) / (x1 - x2)
	b = y1 - a * x1

	generator = build_generator()
	# fig, ax = plt.subplots()

	for i in range(10):
		rand_x = np.random.uniform(x1, x2)
		rand_y = a * rand_x + b

		z_sample = np.array([[rand_x, rand_y]])
		x_decoded = generator.predict(z_sample)
		plt.subplot(2,5,i + 1)
		plt.imshow(x_decoded.reshape(28,28))
		if i == 2:
			plt.title('Generated Images')

	plt.savefig(fig_name)
	plt.show()

def parse_args(argv):
	if len(argv) < 2:
		return 'c-e'

	cmd = "-q [c-e|f]"

	if len(argv) >= 2:
		i = 1
		while i < len(argv):
			if argv[i] == "-q":
				assert (i + 1) < len(argv), "Missing argument. Correct cmd: " + cmd 
				qname = argv[i + 1]
				assert qname in ['c-e', 'f'], \
					"Bad model name: " + argv[i + 1] + " Use c-e|f"
				i += 1
			else:
				assert False, "bad command line. Correct cmd: " + cmd
			i += 1
	return qname

if __name__ == '__main__':
	figure_names = ['QuestionC','QuestionD','QuestionE']
	qname = parse_args(sys.argv)
	if qname == 'f': 
		q_log_var = np.zeros(latent_dim) # question f
		figure_names = ['QuestionF_c','QuestionF_d','QuestionF_e']

	QuestC(figure_names[0])
	QuestD(figure_names[1])
	QuestE(figure_names[2])




