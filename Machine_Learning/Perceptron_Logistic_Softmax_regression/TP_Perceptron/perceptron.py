import numpy as np

class perceptron(object):
   
    def __init__(self, input_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values.
        Weights are stored in the variable self.params, which is a dictionary
        with the following keys:

        w: Weights; has shape (D, 1)

        Inputs:
        - input_size: The dimension D of the input data
        """
        self.params = {}
        self.params['W'] = std * np.random.randn(input_size, 1)

    
    def loss(self, X, y=None):
        """
        Compute the loss and gradients for the perceptron.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
         -1 or 1. This parameter is optional; if it is not passed then we only 
        return scores, and if it is passed then we instead return the loss and 
        gradients.

        Returns:
        If y is None, return a matrix scores of shape (N, 1) where scores[i] is
        the output of perceptron on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss for this batch of training samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
        with respect to the loss function; has the same keys as self.params.
        """
        N, D = X.shape
        
        # Compute the forward pass
        scores = None
        #############################################################################
        # TODO: Perform the forward pass, computing the class scores for the input. #
        # Store the result in the scores variable, which should be an array of      #
        # shape (N, 1) with values 1 or -1                                          #
        #############################################################################
        scores_exactes = X.dot(self.params['W'])
        
        scores = np.sign(scores_exactes)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        
        # If the targets are not given then jump out, we're done
        if y is None:
            return scores

        # Compute the loss
        if scores is not None:
            y = y.reshape(scores.shape)
        loss = None
        #########################################################################
        # TODO: Perform the loss computation. Store the result in loss variable,#
        # which should be a scalar.                                             #
        #########################################################################
        #E(w) = − [Xw]T(y − sign(Xw))
        #Que les mal classés
        
        loss = -scores_exactes.T.dot((y - scores)/2)
        loss = loss[0][0]
        
        #########################################################################
        #                             END OF YOUR CODE                          #
        #########################################################################


        # Backward pass: compute gradients
        grads = {}
        #########################################################################
        # TODO: Perform the grads computation. Store the result in grads dict.  #
        # You should have the same keys in grads than self.params and the shape #
        # of the grads['key'] should be exactly the same as self.params['key']  #
        #########################################################################
        #∇ E(w) = − XT * (y − sign(Xw))
        test = X.T.dot((y-scores))
        grads['W'] = test
        #########################################################################
        #                             END OF YOUR CODE                          #
        #########################################################################

        return loss, grads


    def train(self, X, y, learning_rate=1e-3, num_epochs=1000, batch_size=10, display=False):
        """
        Train the perceptron.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
        X[i] has label c, where c = -1 or c = 1.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - num_epochs: Number of steps to take.
        - batch_size: Number of training examples before update
        - display: Boolean, if true display informations
        """
        # this values will keep a trace of what happens 
        loss_history = []
        train_acc_history = []
        val_acc_history = []


        num_train = X.shape[0]
        training_data = []
        training_labels = []

        #########################################################################
        # TODO: Split the data every batch_size. Put your data splited in       #
        # training_data and split labels the same way and put them in           #
        # training_lebels list                                                  #
        #########################################################################
        #DIVISION de l'ensemble d'entrainement en plusieurs mini-lots : chaque mini-lot contient un sous-ensemble de donnée d'entrainement
        for i in range(0, num_train, batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            training_data.append(X_batch)
            training_labels.append(y_batch)
        pass
        #########################################################################
        #                             END OF YOUR CODE                          #
        #########################################################################

        # if num_epochs == 0 return training_data, training_labels to check if your implementation
        # is correct
        if not num_epochs:
            return training_data, training_labels


        for epoch in range(num_epochs):
            for idx_batch in range(len(training_data)):
                X_batch = training_data[idx_batch]
                y_batch = training_labels[idx_batch]

                # Compute loss and gradients using the current minibatch
                loss, grads = self.loss(X_batch, y=y_batch)
                loss_history.append(loss)

                #########################################################################
                # TODO: Use the gradients in the grads dictionary to update the         #
                # parameters of the network (stored in the dictionary self.params)      #
                # using stochastic gradient descent. You'll need to use the gradients   #
                # stored in the grads dictionary defined above.                         #
                #########################################################################
                self.params['W'] += learning_rate * grads['W']
                #########################################################################
                #                             END OF YOUR CODE                          #
                #########################################################################

                if display:
                    print("iteration {} / {}: loss {}".format(idx_batch + 1 + (epoch*len(training_data)), num_epochs*len(training_data), loss))

                # Every epoch, check train and val accuracy and decay learning rate.
                if idx_batch == len(training_data)-1:
                    # Check accuracy
                    train_acc = (self.predict(X).T == y).mean()
                    train_acc_history.append(train_acc)

        return {
        'loss_history': loss_history,
        'train_acc_history': train_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
        classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
        the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
        to have class c, where 0 <= c < C.
        """
        y_pred = None

        

        ###########################################################################
        # TODO: Implement this function; it should be VERY simple!                #
        ###########################################################################
        scores = X.dot(self.params['W'])
        y_pred = np.sign(scores).flatten()
        ###########################################################################
        #                              END OF YOUR CODE                           #
        ###########################################################################

        return y_pred