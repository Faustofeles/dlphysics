import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing


def load_mnist(flatten=False, one_hot=False, normalize=False):
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    # normalize x
    if normalize:
        X_train = X_train.astype(float) / 255.
        X_test = X_test.astype(float) / 255.
    # we reserve the last 10000 training examples for validation
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]
    
    if flatten:
        X_train = X_train.reshape([X_train.shape[0], -1])
        X_val = X_val.reshape([X_val.shape[0], -1])
        X_test = X_test.reshape([X_test.shape[0], -1])
    
    if one_hot:
        y_train = one_hot_encode(y_train)
        y_val = one_hot_encode(y_val)
        y_test = one_hot_encode(y_test)
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def normalize(X):
    return preprocessing.normalize(X)

def standardize(X):
    return preprocessing.scale(X)

def one_hot_encode(labels):
    oh = np.zeros((len(labels), 10))
    oh[np.arange(len(labels)), labels] = 1
    return oh

###############################
#      PLOTTING RESULTS       #
###############################

'''

'''
def plot_model_results(train_log, val_log, train_loss, val_loss, params=False):
    import seaborn as sns   # make it pretty
    with sns.axes_style("darkgrid"):
        
        fig, axes = plt.subplots(nrows=1, ncols=2)

        

        # accuracy metrics
        axes[0].plot(train_log, label='train accuracy')
        axes[0].plot(val_log, label='val accuracy')
        axes[0].set_title("Train Accuracy: %.3f | Val Accuracy: %.3f"%( train_log[-1], val_log[-1]))
        axes[0].set_ylabel("Accuracy")
        axes[0].set_xlabel("Epoch")
        axes[0].legend(loc='best')

        # loss metrics
        axes[1].plot(train_loss, label='train loss')
        axes[1].plot(val_loss, label='val loss')
        axes[1].set_title("Train Loss: %.5f | Val Loss: %.5f"%(train_loss[-1], val_loss[-1]))
        axes[1].set_ylabel("Loss")
        axes[1].set_xlabel("Epoch")
        axes[1].legend(loc='best')
        plt.show()

        if params:
            # title of figure
            title = \
            "Arch :: Epochs: %s | Batch Size: %s | Learning Rate: %s"\
            %(params)
            fig.suptitle(title)
            # save figure
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