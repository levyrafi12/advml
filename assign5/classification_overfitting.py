# Here we learn a binary classifier using a one-hidden layer network. 
# We experiment with adding random labels to show that overfitting is possible
# (change frac_random to explore)

from __future__ import print_function

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)

def model_fn_nn(features, labels, mode,params):
	d = features['x'].get_shape().as_list()[1]
	train_first_layer = params['train_first_layer']
	n_hidden = params['n_hidden']
	w1 = tf.get_variable('w1', shape=(d,n_hidden), dtype=tf.float64, trainable=train_first_layer)
	b1 = tf.get_variable('b1', shape=(1,n_hidden), dtype=tf.float64, trainable=train_first_layer)
	w2 = tf.get_variable('w2', shape=(n_hidden,1), dtype=tf.float64)

	h1 = tf.nn.leaky_relu(tf.matmul(features['x'],w1) + 4.0*b1)
	predictions = tf.matmul(h1,w2)

	if mode == tf.estimator.ModeKeys.PREDICT: 
		return tf.estimator.EstimatorSpec(mode=mode, predictions={'y': predictions, 'x': features['x']})
	loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(labels, tf.float64), logits = predictions)
	loss = tf.reduce_mean(loss)

	optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
	train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

	acc_op = tf.metrics.accuracy(labels = labels, predictions=predictions>0)

	# TF Estimators requires to return an EstimatorSpec, that specify
	# the different ops for training, evaluating, ...
	estim_specs = tf.estimator.EstimatorSpec(
	mode=mode,
	predictions={'y': predictions, 'x': features['x']},
    loss=loss,
    train_op=train_op,
    eval_metric_ops={'acc': acc_op})

	return estim_specs

def plot_2d_classifier(func):
	xmin = -1
	xmax = 1
	ymin = -1
	ymax = 1
	mesh_step = 0.01
	x = np.arange(xmin, xmax, mesh_step) # 200
	y = np.arange(ymin, ymax, mesh_step)
	xx, yy = np.meshgrid(x, y) # 40,000 * 1
	Z = func(np.c_[xx.ravel(), yy.ravel()]) # 20,000 * 1
	plt.figure()
	Z = np.reshape(Z,xx.shape)
	plt.pcolormesh(xx, yy, np.sign(Z))
	plt.show()

import numpy as np
from matplotlib import pyplot as plt

eps = 0.015

def relu(x):
	return x * (x > 0)

def gt_nn_func(x):
	w1 = np.asarray([[1,1],[1,1],[-1,1],[-1,1]])
	b1 = np.asarray([[0, -eps, 0, -eps]])
	w2 = np.asarray([[1],[-1],[1],[-1]])
	b2 = np.asarray([[-eps]])
	h = relu(np.matmul(x, w1.transpose()) + b1)
	f = np.matmul(h, w2) + b2
	return np.where(np.isclose(f, eps), 1, 0)

def gt_logic_and_func(x):
	w1 = np.asarray([[1],[1]]) # 2 * 1
	w2 = np.asarray([[-1],[1]]) # 2 * 1
	# dim(x) is n * 2 where n = len(x)
	# 1 to all points above the diagonal y = -x, otherwise 0
	mask = np.logical_and(np.matmul(x, w1) > 0, np.matmul(x, w2) > 0)
	return np.reshape(np.where(mask, 1, 0), (-1, 1)) # n * 1

def gt_func(x):
	w = np.asarray([[1],[1]]) # 2 * 1
	# dim(x) is n * 2 where n = len(x)
	# 1 to all points above the diagonal y = -x, otherwise 0
	return np.reshape((np.matmul(x,w)>0).astype(int),(-1,1)) # n * 1

# plot_2d_classifier(gt_func_ptr())

def train_regression(params,X,y, n_train, num_steps):
	batch_size = 32

	train_fn = tf.estimator.inputs.numpy_input_fn(
		x={'x': X[0:n_train]}, y=y[0:n_train],
		batch_size=batch_size, num_epochs=None, shuffle=True)
	eval_fn = tf.estimator.inputs.numpy_input_fn(
		x={'x': X[n_train:-1]}, y=y[n_train:-1],
		batch_size=batch_size, num_epochs=1, shuffle=False)
	eval_train_fn = tf.estimator.inputs.numpy_input_fn(
		x={'x': X[0:n_train]}, y=y[0:n_train],
		batch_size=batch_size, num_epochs=1, shuffle=False)

	model_fn = model_fn_nn

	model = tf.estimator.Estimator(model_fn,params=params)
	model.train(train_fn, steps=num_steps)
	acc_train = model.evaluate(eval_train_fn)['acc']
	acc_test = model.evaluate(eval_fn)['acc']
	d = params['n_hidden']
	print('d: ', d, 'accuracy eval: ', acc_test , 'accuracy train: ', acc_train)
	return model, acc_test, acc_train

def set_data(func, n_train, big_n=500, num_steps=500):
	np.random.seed(seed=0)
	X = np.random.randn(big_n, 2) # 500 * 2
	y = func(X) # 500 * 2
	y = np.hstack([y,]) # 500 * 1
	# Replace some of the training labels with random
	frac_random = 0.00001
	k = round(n_train*(1-frac_random))
	y[k:n_train] = (np.random.randn(n_train-k,1)>0).astype(int) # (n - k) * 1
	return X, y, k

def train_regression_iter(X, y, n_train, hidden_units, num_steps=500):
	models = {}
	acc_test_arr = []
	acc_train_arr = []
	for d in hidden_units:
		params = {'n_hidden': d, 'train_first_layer': True}
		model_name = 'model_train_first'+ str(d)
		models[model_name], acc_test, acc_train = train_regression(params,X,y,n_train,num_steps)
		acc_test_arr.append(acc_test)
		acc_train_arr.append(acc_train)
	return models, acc_test_arr, acc_train_arr

def plot_tf_2d_classifier(model, train_x, train_y, fig_name, noise_starts):
	v = 5
	xmin = -v
	xmax = v
	ymin = -v
	ymax = v
	mesh_step = 0.1
	batch_size = 32
	x1 = np.arange(xmin, xmax, mesh_step) #random.uniform(0,np.pi * 2, [300,1])
	x2 = np.arange(ymin, ymax, mesh_step)
	xx1, xx2 = np.meshgrid(x1, x2) # 100 * 100
	stacked_x = np.c_[xx1.ravel(), xx2.ravel()] # 10,000 * 2
	y = np.zeros((stacked_x.shape[0],1))

	eval_fn = tf.estimator.inputs.numpy_input_fn(
		x={'x': stacked_x}, y=y,
		batch_size=batch_size, num_epochs=1, shuffle=False)
	pred = model.predict(eval_fn)
	ys_test = []
	xs_test = []
	for p in pred:
		ys_test.append(p['y'])
		xs_test.append(p['x'])
	ys_test = np.concatenate(ys_test)
	xs_test = np.concatenate(xs_test)
	Z = ys_test 
	Z = np.reshape(Z,xx1.shape) # 10,000 * 2
	plt.pcolormesh(xx1, xx2, np.sign(Z))
	train_y_clean = train_y[:noise_starts] # dim 60 * 1 
	train_x_clean = train_x[:noise_starts] # dim 60 * 2
	train_y_noise = train_y[noise_starts:]
	train_x_noise = train_x[noise_starts:]
	x0_clean = train_x_clean[np.where(train_y_clean[:,0]==0)]
	x1_clean = train_x_clean[np.where(train_y_clean[:,0]==1)]
	x0_noise = train_x_noise[np.where(train_y_noise[:,0]==0)]
	x1_noise = train_x_noise[np.where(train_y_noise[:,0]==1)]
	plt.scatter(x0_clean[:,0], x0_clean[:,1],c='r')
	plt.scatter(x1_clean[:,0], x1_clean[:,1],c='b')  
	plt.scatter(x0_noise[:,0], x0_noise[:,1],c='m')
	plt.scatter(x1_noise[:,0], x1_noise[:,1],c='g')
	plt.title(fig_name)
	plt.savefig(fig_name)
	# plt.show()

def Quest1a():
	n_train = 60
	X, y, k = set_data(gt_logic_and_func, n_train)
	models, _, _ = train_regression_iter(X, y, n_train, [100])
	for name, model in models.items():
		plot_tf_2d_classifier(model,X[0:n_train], y[0:n_train], "2D_classifer_1a", noise_starts = k)

def Quest1b():
	n_train = 60
	X, y, k = set_data(gt_nn_func, n_train)
	models, _, _ = train_regression_iter(X, y, n_train, [100])
	for name, model in models.items():
		plot_tf_2d_classifier(model,X[0:n_train], y[0:n_train], "2D_classifier_1b", noise_starts = k)

def Quest1c():
	n_train = 50
	X, y, k = set_data(gt_nn_func, n_train)
	d_vals = [1,4,5,10,50,100]
	models, test_acc, train_acc = train_regression_iter(X, y, n_train, d_vals)
	# plt.axis([0, 100, 0, 1.0])
	plt.xlabel('n hidden units')
	plt.ylabel('accuracy')
	plt.plot(d_vals, test_acc, 'r--', linewidth=3, label='test')
	plt.plot(d_vals, train_acc, 'b--', linewidth=3, label='train')
	plt.legend(loc='center right', fontsize='x-large')
	plt.title('acc vs num hidden units')
	plt.savefig('graph_1c')

def Quest1d():
	n_train = 60
	X, y, k = set_data(gt_nn_func, n_train)
	# increase negative examples
	X[:n_train,0] = X[:n_train,0] * 16
	models, _, _ = train_regression_iter(X, y, n_train, [100])
	for name, model in models.items():
		plot_tf_2d_classifier(model,X[0:n_train], y[0:n_train], "2D_classifier_1d", noise_starts = k)

Quest1a()
Quest1b()
Quest1c()
Quest1d()

# Add margin
# Play with nummber of units
# Play with GT function

