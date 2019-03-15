"""
University of Texas at El Paso
Dr. Munoz Group, Department of Physics

#### Classification of handwritten numbers ####

Concepts:
    - Might need to download: Keras and TQDM libraries
	- MNIST Dataset [70K examples of 28 by 28 pixel images]
	- Normalization of Data
	- MSE Cost Function
	- One Hot Encoding
	- Dataset Pieces: Training Set, Testing Set, and Validation Test
	- Model: Dense Layer, ReLU Layer
	- Activation
	- Hyperparameters
	- Mini-batches and Stochastic Gradient Descent
	- Epochs
	- Accuracy and Loss Graphs
	- Confusion Matrix
"""

from __future__ import print_function
import keras								## Tool to download dataset
import numpy as np 							## For numerical python
from tqdm import trange						## Progress bar tool
import matplotlib.pyplot as plt 			## Plotting library
# from IPython.display import clear_output	## Interactive Spyder	

np.random.seed(42)							## Random Seed: allows experiment replication


#############################################
#				MNIST DATASET 				# 
#############################################

'''
	Function uses the Keras library to load the MNIST Dataset, a 
	dataset of 70000 training examples of handwritten digits  

	Returns training set [50000 examples], validation set [10000 samples], 
	and test sets [10000 samples]
'''
def load_dataset(flatten=False):
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalization of training data
    X_train = X_train.astype(float) / 255.
    X_test = X_test.astype(float) / 255.
    
    # We reserve the last 10000 training examples for validation
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]
    
    if flatten:
        X_train = X_train.reshape([X_train.shape[0], -1])
        X_val = X_val.reshape([X_val.shape[0], -1])
        X_test = X_test.reshape([X_test.shape[0], -1])
    
    return X_train, y_train, X_val, y_val, X_test, y_test

#############################################
#		MULTILAYER PERCEPTRON BLOCKS  		# 
#############################################

'''
	This is a Layer class that serves as a general "recipe"/blueprint 
	structure (parent) for different types of Layer Objects. 
'''
class Layer:
    def __init__(self):
        pass

    def forward(self, input):
        return input

    def backward(self, input, grad_output):
        num_units = input.shape[1]
        d_layer_d_input = np.eye(num_units)
        return np.dot(grad_output, d_layer_d_input) # chain rule

'''

'''
class Dense(Layer):
    def __init__(self, input_units, output_units, learning_rate=0.1):
        # layer which performs a learned affine transformation:
        # f(x) = <W*x> + b
        self.learning_rate = learning_rate
        self.weights = np.random.normal(loc=0.0, 
                                        scale = np.sqrt(2/(input_units+output_units)), 
                                        size = (input_units,output_units))
        self.biases = np.zeros(output_units)
        
    def forward(self,input):
        # Perform an affine transformation:
        # f(x) = <W*x> + b
        
        # input shape: [batch, input_units]
        # output shape: [batch, output units]
        
        return np.dot(input,self.weights) + self.biases
    
    def backward(self,input,grad_output):
        # compute d f / d x = d f / d dense * d dense / d x
        # where d dense/ d x = weights transposed
        grad_input = np.dot(grad_output, self.weights.T)
        
        # compute gradient w.r.t. weights and biases
        grad_weights = np.dot(input.T, grad_output)
        grad_biases = grad_output.mean(axis=0)*input.shape[0]
        
        assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape
        
        # Here we perform a stochastic gradient descent step. 
        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases
        
        return grad_input

'''

'''
class ReLU(Layer):
    def __init__(self):
        pass
    
    def forward(self, input):
        # Apply elementwise ReLU to [batch, input_units] matrix
        relu_forward = np.maximum(0,input)
        return relu_forward
    
    def backward(self, input, grad_output):
        # Compute gradient of loss w.r.t. ReLU input
        relu_grad = input > 0
        return grad_output*relu_grad


#############################################
#			LOSS FUNCTION: MSE 				# 
#############################################

def one_hot_encode(labels):
    oh = np.zeros((len(labels), 10))
    oh[np.arange(len(labels)), labels] = 1
    return oh

def one_hot_decode(labels):
    return np.argmax(labels, axis=1)

def mean_square_error(outputs, y):
    #pred_oh = one_hot_encode(pred)
    y_oh = one_hot_encode(y)
    return np.sum(np.power(np.subtract(outputs, y_oh), 2), axis=-1)

def grad_mean_square_error(outputs, y):
    y_oh = one_hot_encode(y)
    grad = 2*np.subtract(outputs, y_oh)/outputs.shape[0]
    return grad

#############################################
#		MULTILAYER PERCEPTRON FUNCTIONS		# 
#############################################

def forward(network, X):
    # Compute activations of all network layers by applying them sequentially.
    # Return a list of activations for each layer. 
    
    activations = []
    input = X
    # Looping through each layer
    for l in network:
        activations.append(l.forward(input))
        # Updating input to last layer output
        input = activations[-1]
    
    assert len(activations) == len(network)
    return activations

def predict(network,X):
    # Compute network predictions. Returning indices of largest Logit probability
    outputs = forward(network,X)[-1]
    # print("outputs predict:", outputs.shape)
    return outputs.argmax(axis=-1)

def train(network,X,y):
    # Train our network on a given batch of X and y.
    # We first need to run forward to get all layer activations.
    # Then we can run layer.backward going from last to first layer.
    # After we have called backward for all layers, all Dense layers have already made one gradient step.
    
    # Get the layer activations
    layer_activations = forward(network,X)
    layer_inputs = [X]+layer_activations  #layer_input[i] is an input for network[i]
    outputs = layer_activations[-1]
    
    loss = mean_square_error(outputs, y)
    loss_grad = grad_mean_square_error(outputs, y)

    #### BACKPROPAGATION LOOP ####
    # Propagate gradients through the network
    # Reverse propogation as this is backprop
    for layer_index in range(len(network))[::-1]:
        layer = network[layer_index]
        loss_grad = layer.backward(layer_inputs[layer_index],loss_grad) #grad w.r.t. input, also weight updates
        
    return np.mean(loss)


###############################
# 		MODEL CREATION 		  #
###############################

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.random.permutation(len(inputs))
    for start_idx in trange(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# load the dataset # 
X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(flatten=True)

'''
	Logs: we want to keep track of the accuracy and loss in our
	model at each step in our training process
'''  
train_acc_log = []			# keeps track of training accuracy  
val_acc_log = []            # keeps track of validation accuracy
train_loss_log = []			# keeps track of training loss
val_loss_log = []			# keeps track of training loss


# Visualize the dataset:

# plt.figure(figsize=[6,6])
# for i in range(4):
#     plt.subplot(2,2,i+1)
#     plt.title("Label: %i"%y_train[i])
#     plt.imshow(X_train[i].reshape([28,28]),cmap='gray');


# Hyperparameters (simplest hyperparameters) #
epochs = 5
batchsize = 32
lr = 0.1

# Creating the MLP Network Architecture
network = []
network.append(Dense(X_train.shape[1],100, learning_rate=lr))
network.append(ReLU())
network.append(Dense(100,200, learning_rate=lr))
network.append(ReLU())
network.append(Dense(200,10, learning_rate=lr))


###############################
# 	 THE ACTUAL TRAINING 	  #
###############################

for epoch in range(epochs): # One epoch is a full run through the whole training set (forward and backward pass)
    for x_batch,y_batch in iterate_minibatches(X_train, y_train, batchsize=batchsize, shuffle=True):
        train(network,x_batch,y_batch) # one forward and backward pass for a given batch

    # Log accuracy and loss in each epoch #
    # Predictions #
    pred_train = predict(network, X_train)
    pred_val = predict(network, X_val)

    # Metrics #
    train_acc = np.mean(pred_train == y_train)
    val_acc = np.mean(pred_val == y_val)
    train_loss = np.mean(mean_square_error(one_hot_encode(pred_train), y_train)) # ask me about this 
    val_loss = np.mean(mean_square_error(one_hot_encode(pred_val), y_val))       # ask me about this

    # Log #
    train_acc_log.append(train_acc)     # accuracy
    val_acc_log.append(val_acc)

    train_loss_log.append(train_loss)   # loss
    val_loss_log.append(val_loss)

    # clear_output()
    # Display metrics of each epoch #
    print("Epoch %s:"%str(epoch+1))
    print("Train accuracy:",train_acc_log[-1])
    print("Validation accuracy:",val_acc_log[-1])
    print("Train Loss:", train_loss_log[-1])
    print("Validation Loss:", val_loss_log[-1])
    

###############################
#      PLOTTING RESULTS       #
###############################

'''

'''
def plot_model_results(train_log, val_log, train_loss, val_loss):
    import seaborn as sns   # make it pretty
    with sns.axes_style("darkgrid"):
        
        fig, axes = plt.subplots(nrows=1, ncols=2)

        # title of figure
        title = \
        "Arch :: Epochs: %s | Batch Size: %s | Learning Rate: %s"\
        %(epochs, batchsize, lr)
        fig.suptitle(title)

        # accuracy metrics
        axes[0].plot(train_log, label='train accuracy')
        axes[0].plot(val_log, label='val accuracy')
        axes[0].set_title("Train Accuracy: %.3f"%train_log[-1])
        axes[0].set_ylabel("Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].legend(loc='best')

        # loss metrics
        axes[1].plot(train_loss, label='train loss')
        axes[1].plot(val_loss, label='val loss')
        axes[1].set_title("Train Loss: %.5f"%train_loss[-1])
        axes[1].set_ylabel("Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].legend(loc='best')
        plt.show()

        # save figure
        params = epochs, batchsize, str(lr).replace('.', ',')
        fig.savefig("mlp_e%s_b%s_lr%s.png"%(params))

'''
    function taken from: 
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
'''
from sklearn.metrics import confusion_matrix
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


plotting accuracy and loss graphs #

plot_model_results(train_acc_log, val_acc_log, 
                   train_loss_log, val_loss_log)


confusion matrix #
class_names = [0,1,2,3,4,5,6,7,8,9]
pred_test = predict(network, X_test)   # using test set (finally!)
plot_confusion_matrix(y_test, pred_test, classes=class_names, title="Confusion Matrix")
plt.show()
