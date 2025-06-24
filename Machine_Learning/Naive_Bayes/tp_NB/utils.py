import numpy as np
from sklearn import datasets
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


def load_IRIS(test=True):
    iris = datasets.load_iris()
    X, y = shuffle(iris.data, iris.target, random_state= 1234)
    #iris.target represents the 3 possible classes of iris
    if test:
        X_train = X[:100, :]
        y_train = y[:100]
        X_test = X[100:, :]
        y_test = y[100:]
        return X_train, y_train, X_test, y_test
    else:
        return X, y


def train_test_split(X, y, test_size=0.3, normalize = False):
    """ Split the data into train and test sets """

    # Split the training data from test data in the ratio specified in
    # test_size
    split_i = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]
    if normalize == True:
        mean_train = np.mean(X_train)
        std_train = np.std(X_train)
        X_train = (X_train - mean_train) / std_train
        X_test = (X_test - mean_train) / std_train
        #min_train =  np.min(X_train)
        #max_train = np.max(X_train)
        #X_train = (X_train - max_train) / (max_train- min_train)
        #X_test = (X_test - max_train) / (max_train- min_train)

    return X_train, y_train, X_test, y_test

def plot_decision_boundary(model, X, y, axes_label_1, axes_label_2):
    model.train(X,y)

    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5

    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict the function value for the whole gid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = np.array(Z).reshape(xx.shape)

    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.xlabel(axes_label_1)
    plt.ylabel(axes_label_2)
    # Plot the decision boundary
    # create scatter plot for samples from each class
    labels = []
    for class_value in range(len(np.unique(y))):
        # get row indexes for samples with this class
        row_ix = np.where(y == class_value)
        # create scatter of these samples

        plt.scatter(X[row_ix, 0], X[row_ix, 1], s=50, cmap=plt.cm.Spectral, label= class_value)#, color=colors_points[class_value])
    # show the plot
    plt.legend(loc='upper left')
    plt.show()



def compute_accuracy(y_true, y_pred):
    """
    Returns the classification accuracy.
    Inputs:
    ----------
    y_true : True labels for X
    y_pred : Predicted labels for X

    Returns
    -------
    accuracy : float
    """
    ##################################################################
    # ToDo: compute the accuracy among the true and predicted labels #
    # use only numpy functions                                       #
    ##################################################################
    accuracy = np.mean(y_true == y_pred)
    return accuracy

