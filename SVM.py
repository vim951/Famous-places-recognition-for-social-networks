from __future__ import print_function
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import pandas as pd
from builtins import range
from database import load_db_csv , id_to_np, joined_shuffle

from builtins import object

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')
print("Files imported successfully")
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 30 * 30 + 1)

## Constants

csv_db_path = '/Users/hugodanet/Downloads/train_clean.csv'
csv_labels_path = '/Users/hugodanet/Downloads/train_label_to_category.csv'
preprocessed_db_path = '/Users/hugodanet/Downloads/PDB 2'

train_size = 36966
size = 100
number_of_classes = 50

##DATA PREPARATION

classes_list = []
for i in range(number_of_classes):
    classes_list.append(i)

C,L = load_db_csv(number_of_classes)
X,Y,W=[],[],[]

for i in range(number_of_classes):
    for x in C[i][1].split(' '):
        if not id_to_np(x) is None:
            X.append(id_to_np(x))
            Y.append([i])
            W.append(i)
        
X,Y = joined_shuffle(X, Y)

class_weights = class_weight.compute_class_weight('balanced',np.unique(W),W)

class_weight_dict = dict(enumerate(class_weights))

Xarr = np.array(X)
Yarr = np.array(Y)

X_train, X_test, y_train, y_test = train_test_split(Xarr, Yarr, random_state=42, test_size=0.20)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, random_state=42, test_size=0.5)

print("X_train: " + str(X_train.shape))
print("X_test: " + str(X_test.shape))
print("X_val: " + str(X_val.shape))
print("y_train: " + str(y_train.shape))
print("y_test: " + str(y_test.shape))
print("y_val: " + str(y_val.shape))

####Forming X_test, X_train, y_train, y_test####
num_training = X_train.shape[0]
mask = list(range(num_training))
X_train = X_train[mask]
y_train = y_train[mask]
num_test = X_test.shape[0]
mask = list(range(num_test))
X_test = X_test[mask]
y_test = y_test[mask]
num_val = X_val.shape[0]
mask = list(range(num_val))
X_val = X_val[mask]
y_val = y_val[mask]

# Reshape the image data into rows
X_train = np.reshape(X_train, (X_train.shape[0], -1))
X_test = np.reshape(X_test, (X_test.shape[0], -1))
X_val = np.reshape(X_val, (X_val.shape[0], -1))
print(X_train.shape, X_test.shape, X_val.shape)
print(y_train.shape, y_test.shape, y_val.shape)

# Getting data to zero mean, i.e centred around zero.
mean_image = np.mean(X_train, axis=0)
X_train -= mean_image
X_test -= mean_image
X_val -= mean_image
# append the bias dimension of ones (i.e. bias trick) so that our   # SVM
# only has to worry about optimizing a single weight matrix W.
X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
print(X_train.shape, X_test.shape, X_val.shape)
print("Data ready")

def svm_loss_vectorized(W, X, y, reg):
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    num_classes = W.shape[1]
    num_train = X.shape[0]
    scores = X.dot(W)
    y = [int(x) for x in y]
    correct_class_scores = scores[np.arange(num_train), y].reshape(num_train, 1)
    margin = np.maximum(0, scores - correct_class_scores + 1)
    margin[np.arange(num_train), y] = 0  # do not consider correct class in loss
    loss = margin.sum() / num_train
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    margin[margin > 0] = 1
    valid_margin = margin.sum(axis=1)
    margin[np.arange(num_train), y] -= valid_margin
    dW = (X.T).dot(margin) / num_train
    # Regularization gradient
    dW = dW + reg * 2 * W
    return loss, dW

class LinearClassifier(object):

    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):

        num_train, dim = X.shape
        num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, int(num_classes))
        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):
            X_batch = None
            y_batch = None

            batch_indices = np.random.choice(num_train, batch_size, replace=False)
            X_batch = X[batch_indices]
            y_batch=y[batch_indices]

            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # Update the weights using the gradient and the learning rate.          #

            self.W -= learning_rate*grad

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        y_pred = np.zeros(X.shape[0])
        scores = X.dot(self.W)
        y_pred = scores.argmax(axis=1)
        return y_pred


class LinearSVM(LinearClassifier):
    """ A subclass that uses the Multiclass SVM loss function """

    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)

svmd = LinearSVM()

for i in range (0,100):
    loss_hist = svmd.train(X_train, y_train, learning_rate=1e-7, reg=2.5e4, num_iters=1500, verbose=True)

    y_train_pred = svmd.predict(X_train)
    print('training accuracy: %f' % (np.mean(y_train == y_train_pred),))
    y_val_pred = svmd.predict(X_val)
    print('validation accuracy: %f' % (np.mean(y_val == y_val_pred),))
