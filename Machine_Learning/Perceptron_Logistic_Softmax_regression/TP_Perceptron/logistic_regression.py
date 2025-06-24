import numpy as np
from random import shuffle
from classifier import Classifier


class Logistic(Classifier):
    """A subclass of Classifier that uses the logistic function to classify."""
    def __init__(self, random_seed=0):
        super().__init__('logistic')
        if random_seed:
            np.random.seed(random_seed)



    def loss(self, X, y=None, reg=0):
        """
        Softmax loss function, vectorized version.

        Inputs and outputs are the same as softmax_loss_naive.
        """
        # Initialize the loss and gradient to zero.
        scores = None
        loss = None
        dW = np.zeros_like(self.W)
        num_classes = self.W.shape[1]
        num_train = X.shape[0]
        """
        print(X.shape)
        print(y.shape)
        print(self.W.shape)
        print(scores.shape)
        """
        
        #scores
        #############################################################################
        # TODO: Compute the scores and store them in scores.                        #
        #############################################################################
        #Produit scalaire des instances dans X avec les poids du model
        scores = np.dot(X,self.W)

        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################
        if y is None:
            return scores

        
        # loss
        #############################################################################
        # TODO: Compute the logistic loss and store the loss in loss.               #
        # If you are not careful here, it is easy to run into numeric instability.  #
        # Don't forget the regularization!                                          #
        #############################################################################
        #Calcul de la perte en utilisant la formule : 
        """
                    N 
        Loss= −1/N  ∑ [yi log(pi) + (1 − yi) log(1 − pi)] +
                   i=i 
        N = Nombre d'échantillons
        yi = le label de l'instance i
        pi = la probabilité prédite de l'instance i appartenant à la première classe
        
        pi = 1 / (1+exp(−scorei)) = SIGMOIDE

        """
        # .ravel() transforme scores en un tableau 1D, de sorte que logits contient les scores pour chaque échantillon.
       
        # Calculate the predicted probabilities
        scores = scores.ravel()
        sigmoide = 1 / (1 + np.exp(-scores))
        
        """
        print("sigmoide")
        print(sigmoide, sigmoide.shape)
        """
        # Compute the logistic loss
        EPSILON = 1e-15
        loss = np.mean(-y * np.log(np.clip(sigmoide, EPSILON, 1. - EPSILON)) - (1 - y) * np.log(np.clip(1 - sigmoide, EPSILON, 1. - EPSILON)))
        #loss = np.mean(-y * np.log(sigmoide) - 1*(1 - y) * np.log(1 - sigmoide))

        loss += reg * np.sum(self.W * self.W)
        
        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################

        # grad
        #############################################################################
        # TODO: Compute the gradients and store the gradients in dW.                #
        # Don't forget the regularization!                                          #
        #############################################################################     
        #Formule du gradient
        """
        -> 1/N X.T(p-y) (on connait p qui a été défini avant)
        
        """
      
        sigmoide = 1 / (1 + np.exp(-scores))
        dscores = sigmoide - y[:,None]
        dscores = (sigmoide - y).reshape(-1, 1)
        """
        print("dscores")
        print(dscores, dscores.shape)
        """
        #Gradient
        dW = (X.T.dot(dscores)) / num_train
        """
        print("Gradient")
        print(dW, dW.shape)
        print(np.sum(dW))
        """
        
        #Ajout de la regularization
        dW += reg * 2 * self.W
        """
        print("avec Regularization")
        print(dW, dW.shape)
        print(np.sum(dW))
        """
        
        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################
        return loss, dW

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        """
        scores = X.dot(self.W)
        y_pred = (scores > 0).astype(int)
        """
        new_X = np.dot(X, self.W)
        # print(new_X)
        prob_of_one = 1/(1 + np.exp(-new_X))
        # print(prob_of_one)
        y_pred = np.where(prob_of_one > 0.5, 1, 0)
        # print(y_pred)
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

