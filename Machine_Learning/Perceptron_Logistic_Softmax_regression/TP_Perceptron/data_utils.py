from __future__ import print_function

import os
import platform

import numpy as np
from imageio import imread
from six.moves import cPickle as pickle
from sklearn import datasets
from sklearn.utils import shuffle


def load_IRIS(test=True):
    iris = datasets.load_iris()
    X, y = shuffle(iris.data, iris.target, random_state=1230)
    if test:
        X_train = X[:100, :]
        y_train = y[:100]
        X_test = X[100:, :]
        y_test = y[100:]
        return X_train, y_train, X_test, y_test
    else:
        X = iris.data[:, :]
        y = iris.target
        return X, y


def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == "2":
        return pickle.load(f)
    elif version[0] == "3":
        return pickle.load(f, encoding="latin1")
    raise ValueError("invalid python version: {}".format(version))


def load_CIFAR_batch(filename):
    """load single batch of cifar"""
    with open(filename, "rb") as f:
        datadict = load_pickle(f)
        X = datadict["data"]
        Y = datadict["labels"]
        X = X.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        Y = np.array(Y)
    return X, Y


def load_CIFAR10(ROOT, classes=None):
    """load all of cifar"""
    classes_all = [
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    xs = []
    ys = []
    for b in range(1, 6):
        f = os.path.join(ROOT, "data_batch_%d" % (b,))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y

    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, "test_batch"))
    if classes:
        idxs_tr = []
        idxs_te = []
        for clss in classes:
            idx = classes_all.index(clss)

            idxs_tr += np.flatnonzero(Ytr == idx).tolist()
            idxs_te += np.flatnonzero(Yte == idx).tolist()

        shuffle(idxs_tr)
        shuffle(idxs_te)

        Xtr = Xtr[idxs_tr]
        Ytr = Ytr[idxs_tr]
        Xte = Xte[idxs_te]
        Yte = Yte[idxs_te]

        idx_all_tr = []
        idx_all_te = []
        for clss in classes:
            idx = classes_all.index(clss)

            idx_all_tr.append(np.flatnonzero(Ytr == idx).tolist())
            idx_all_te.append(np.flatnonzero(Yte == idx).tolist())

        for i in range(len(idx_all_tr)):
            Ytr[idx_all_tr[i]] = i
            Yte[idx_all_te[i]] = i

    Xtr, Ytr = shuffle(Xtr, Ytr, random_state=13)
    Xte, Yte = shuffle(Xte, Yte, random_state=13)
    return Xtr, Ytr, Xte, Yte


def get_CIFAR10_data(
    num_training=49000, num_validation=1000, num_test=1000, subtract_mean=True
):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = "cs231n/datasets/cifar-10-batches-py"
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    if subtract_mean:
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_val -= mean_image
        X_test -= mean_image

    # Transpose so that channels come first
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()

    # Package data into a dictionary
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }
