import keras
import numpy as np

from helper_functions import * # plot_confusion_matrix, plot_model_results, load_mnist
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score

mnist = load_mnist(flatten=True, one_hot=True, normalize=True)
X_train, y_train, X_val, y_val, X_test, y_test = mnist

# set random seed for replication
tf.set_random_seed(42)

# defining dimensions of network
n_inputs = 784
n_hidden_1 = 256
n_hidden_2 = 256
n_classes = 10

# defining hyperparameters
learning_rate = 0.001      # how fast it descends the gradient
keep_prob = 1            # num of neuros that will activate
epochs = 15                # num of forward & backward passes through whole training set
batchsize = 128            # mini partitions of training set we train at a time
display_results = True

### Tensor Variables ###

# tensor placeholders for inputs and outputs
X = tf.placeholder("float", [None, n_inputs])  # input dimensions 
Y = tf.placeholder("float", [None, n_classes]) # output dimensions
dropout = tf.placeholder(tf.float32) # dropout regularization (input)

# tensor variables for weights and biases 
# weights are standardized to have mean = 0 and variance = 1
weights = {
	"w0" : tf.Variable(tf.random_normal([n_inputs, n_hidden_1], stddev=(1/tf.sqrt(float(n_inputs))))),
	"w1" : tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2], stddev=(1/tf.sqrt(float(n_hidden_1))))),
	"w2" : tf.Variable(tf.random_normal([n_hidden_2, n_classes], stddev=(1/tf.sqrt(float(n_hidden_2)))))
}

biases = {
	"b0" : tf.Variable(tf.random_normal([n_hidden_1])),
	"b1" : tf.Variable(tf.random_normal([n_hidden_2])),
	"b2" : tf.Variable(tf.random_normal([n_classes]))
}

### MODEL CREATION ###

# define feedforward operation
def forwardpass(X):
	h1 = tf.add(tf.matmul(X, weights["w0"]), biases["b0"]) # ((W*X) + b)
	h1_act = tf.nn.dropout(tf.nn.relu(h1), keep_prob)     # activations
	
	h2 = tf.add(tf.matmul(h1_act, weights["w1"]), biases["b1"])
	h2_act = tf.nn.dropout(tf.nn.relu(h2), keep_prob)     # activations
	
	output = tf.add(tf.matmul(h2_act, weights["w2"]), biases["b2"])
	output_act = tf.sigmoid(output)  # activations 
	return output_act

# graph operations
logits = forwardpass(X)
# loss_op = tf.reduce_mean(tf.losses.mean_squared_error(predictions=fpass, labels=Y))
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

### METRICS LOGS ###
train_acc = []
train_loss = []
val_acc = []
val_loss = []

# def accuracy(actual, pred):
#     correct_prediction = tf.equal(tf.argmax(actual, 1), tf.argmax(pred, 1))
#     acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     return acc

correct_prediction = tf.equal(tf.argmax(y_train, 1), tf.argmax(logits, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # cast a tensor to type float32

### TRAINING MODEL ###
# initializer operator
init = tf.global_variables_initializer()

# initialize a Session to run the Computation Graph
with tf.Session() as sess:
	# initialize model's global variables
	sess.run(init)
	
	for e in range(epochs):
		arr = np.arange(X_train.shape[0])
		np.random.shuffle(arr) # shuffled indices
		for i in range(0, X_train.shape[0], batchsize):
			sess.run(train_op, {X: X_train[arr[i:i+batchsize]],
							 Y: y_train[arr[i:i+batchsize]],
							 dropout:keep_prob})
	
		### RECORD METRICS AT EACH EPOCH ###
		# NOTE: The following is bad, TF has log functions for this
		# 		that we will get to see later in the workshop
		
		# training metrics 
		train_acc.append(sess.run(accuracy, 
						feed_dict={X:X_train,
									Y: y_train,
									dropout: 1}))

		train_loss.append(sess.run(loss_op, 
						feed_dict={X: X_train, 
									Y: y_train,
									dropout:1}))

		# validation metrics 
		val_acc.append(accuracy_score(y_val.argmax(1), 
						sess.run(logits, feed_dict={
							X:X_val, dropout: 1}).argmax(1)))
		
		val_loss.append(sess.run(loss_op, 
						feed_dict={X: X_val, 
								   Y: y_val,
								   dropout:1}))
		
		print("Epoch:{0}, Train loss: {1:.5f} Train acc: {2:.3f}, Val loss: {3:.5f} Val acc:{4:.3f}".format(e+1,
																	train_loss[-1],train_acc[-1], val_loss[-1],val_acc[-1]))

	# predictions for confusion matrix
	pred = sess.run(logits, feed_dict={X:X_test, dropout: 1}).argmax(1) #  predictions for confusion matrix


### DISPLAY RESULTS ###

if display_results:
	# # plotting accuracy and loss graphs #
	params = epochs, batchsize, str(learning_rate).replace('.', ',')
	plot_model_results(train_acc, val_acc, train_loss, val_loss, params=params)
	
	class_names = [0,1,2,3,4,5,6,7,8,9]
	y = np.argmax(y_test, axis=-1)
	acc = accuracy_score(y, pred)
	plot_confusion_matrix(y, pred, classes=class_names, title="Confusion Matrix, Test Accuracy: %s"%acc)
	plt.show()