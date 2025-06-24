import numpy as np
from random import shuffle
from classifier import Classifier


class Softmax(Classifier):
    """A subclass of Classifier that uses the Softmax to classify."""
    def __init__(self, random_seed=0):
        super().__init__('softmax')
        if random_seed:
            np.random.seed(random_seed)

    def loss(self, X, y=None, reg=0):
        scores = None
        # Initialize the loss and gradient to zero.
        loss = 0.0
        dW = np.zeros_like(self.W)
        num_classes = self.W.shape[1]
        num_train = X.shape[0]
        #scores
        #############################################################################
        # TODO: Compute the scores and store them in scores.                        #
        #############################################################################
        #Nous calculons les scores pour chaque point de données en entrée et chaque classe en prenant 
        #un produit scalaire entre les données et la matrice de poids.
        
        scores = X.dot(self.W)


        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################

        if y is None:
            return scores

        # loss
        #############################################################################
        # TODO: Compute the softmax loss and store the loss in loss.                #
        # If you are not careful here, it is easy to run into numeric instability.  #
        # Don't forget the regularization!                                          #
        #############################################################################
        """
        Nous décalons d'abord les scores pour assurer une stabilité numérique. 
        Cela est dû au fait que la fonction exponentielle peut augmenter très rapidement pour des entrées relativement petites.
        Nous calculons les probabilités de classe en utilisant la fonction softmax.
        Nous calculons ensuite la log-vraisemblance négative des vrais labels de classe. Cela nous donne la perte due aux données.
        Nous ajoutons ensuite une régularisation à la perte. Cela encourage les poids à être petits et aide à éviter le surapprentissage.
        """
        shift_scores = scores - np.max(scores, axis=1).reshape(-1, 1)
        softmax_output = np.exp(shift_scores) / np.sum(np.exp(shift_scores), axis=1).reshape(-1, 1)
        loss = -np.sum(np.log(softmax_output[np.arange(num_train), y]))
        
         # average and add regularization
        loss /= num_train
        loss += reg * np.sum(self.W * self.W)
        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################
        
        # grad
        #############################################################################
        # TODO: Compute the gradients and store the gradients in dW.                #
        # Don't forget the regularization!                                          #
        #############################################################################     
        """
        Nous commençons par le gradient de la fonction softmax.
        Nous soustrayons un des scores des classes correctes.
        Nous moyennons ces scores modifiés sur tous les exemples d'entraînement.
        Ce résultat est ensuite utilisé pour calculer le gradient de la perte par rapport aux poids.
        Nous ajoutons le gradient dû à la régularisation.
        """
        # gradient
        dscores = softmax_output.copy()
        dscores[np.arange(num_train), y] -= 1
        dscores /= num_train
        dW = X.T.dot(dscores)
        dW += 2 * reg * self.W
        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################

        return loss, dW


    def predict(self, X):
        y_pred = np.zeros(X.shape[1])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        """
        Nous calculons les scores pour chaque point de données en entrée et chaque classe.
        Nous prédisons que le label de classe est la classe ayant le score le plus élevé pour chaque point de données.
        """
        scores = X.dot(self.W)
        y_pred = np.argmax(scores, axis=1)

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

