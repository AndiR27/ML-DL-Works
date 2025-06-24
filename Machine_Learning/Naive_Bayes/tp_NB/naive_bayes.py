

import numpy as np
import math



class NaiveBayes():
    """
    Naive Bayes classifier for Categorical and Gaussian data.
    It works with
        - continuous dataset (continuous_features = 'all')
        - categorical dataset (continuous_features = None (default))
        - dataset with both continuous and categorical features
          (continuous_features = [1,3,5,6], specify columns with continuous data)


    Parameters
    ----------
    continuous_features : array-like shape (num_continuous_classes,), columns which have continuous features, ex: continuous_features = [2,5]
                          or
                         'all' (when the dataset contains only continuous data)
                          default=None (when the dataset contains only categorical data)

    """

    def __init__(self, continuous_features=None):

        self.continuous_features = continuous_features

        self.prior = 0
        self.num_categories_each_feature = 0
        self.categorical_features = []
        self.mean = []
        self.var = []
        self.frequencies = []

    def train(self, X, y):
        """Train  Naive Bayes

        Parameters
        ----------
        X : array-like, shape (num_samples, n_features)
            Training vectors, where num_samples is the number of samples
            and n_features is the number of features.
        y : array-like, shape (num_samples,)
            Target values.

        Returns
        -------
        self : object
        """

        if type(X) and type(y) is not np.array:
            X, y = np.array(X), np.array(y)


        # unique target's classes (labels)
        self.classes = np.unique(y)


        (num_samples, num_features) = X.shape

        if self.continuous_features is None:
            # the data set has only categorical features
            self.continuous_features = []
        if self.continuous_features is 'all':
            # the data set has only continuous features
            self.continuous_features = np.arange(0, num_features)

        # Get the index columns of the categorical and continuous data
        self.continuous_features = np.array(self.continuous_features).astype(int)
        self.categorical_features = np.delete(np.arange(num_features), self.continuous_features)

        if self.categorical_features.size != 0:
            
            ########################################################################
            # ToDo: Calculate the number of categories in each categorical_feature #
            # Hint: np.unique,  astype(int)                                        #
            # You cannot use if statement                                          #
            ########################################################################
            """
            On compte le nombre de valeurs uniques pour chaque variable categorical
            """
            #X_categorical = X[:, self.categorical_features].astype(int)
            
            self.num_categories_each_feature = np.array([len(np.unique(X[:, i]).astype(int)) for i in self.categorical_features])

        if self.continuous_features.size != 0:
            
            ####################################################################################
            # ToDo: Compute conditional mean and conditional variance for continuous features  #
            # shape: (num_classes, len(continuous_features))
            # You cannot use if statement                                                      #
            ####################################################################################
            """
            Calcul des moyennes et variances conditionnelles pour chaque variable continue
            """
            X_train_cont = X[:, self.continuous_features]
            self.mean = np.array([np.mean(X[y == k][:, self.continuous_features], axis=0) for k in range(len(self.classes))])
            self.var = np.array([np.var(X[y == k][:, self.continuous_features], axis=0) for k in range(len(self.classes))])
            

        if self.categorical_features.size != 0:
            
            #######################################################
            # ToDo: Fill the  _calculate_frequencies() function   #
            # You cannot use if statement                         #
            #######################################################
            """
            calcule des fréquences conditionnelles pour chaque catégorie de chaque variable catégorical
            """
            
            self.frequencies = self._calculate_frequencies(X,y)


        # calculate prior
        #################################################
        # ToDo: Fill the  _calculate_prior() function   #
        # You cannot use if statement                   #
        #################################################

        self.priors = self._calculate_prior(y)

        return self

    def _calculate_prior(self, y):
        """ Calculate the priors
        (samples where class == c / total number of samples)
        hint: chack the np.bincount() function
        _____________________
        Input:
        - y : array, shape (num_samples,) containing the target values
        Output:
        - priors : array, shape (num_classes,)
        """
        priors = np.bincount(y) / len(y)
        return priors

    def _calculate_frequencies(self, X, y):
        """ Calculate the conditional prob of each value v_jk of x_j for given class y_i = c, P(x_j = v_jk | y_i)
        using relative frequencies.
        - P(x_j = v_jk | y_i) = n_ijk / n_i
            - n_ijk : number of examples in y_i class where x_j = v_jk
            - n_i   : number of examples in y_i class)

            Hint: np.bincount with minlength
        """
        """
        La méthode _calculate_frequencies() renvoie les fréquences conditionnelles pour chaque catégorie de chaque variable de 
        caractéristiques.
        """

        #  empty array
        
       #frequencies = [ np.zeros((self.classes.size, num_categories))for num_categories in self.num_categories_each_feature]
       
        #print("Données de type : Categorical....")
        num_classes = self.classes.size
        #print(num_classes)
        num_features = self.num_categories_each_feature.size
        #print(num_features)
        num_categories = self.num_categories_each_feature.max()
        #print(num_categories)
        frequencies = np.zeros((num_classes, num_features, num_categories))
        #print(frequencies.shape)

        for k in range(num_classes):
            Xk = X[y == k, :]
            n_i = Xk.shape[0]
            for j, feature in enumerate(self.categorical_features):
                n_ijk = np.bincount(Xk[:, feature].astype(int), minlength=self.num_categories_each_feature[j])
                frequencies[k, j, :n_ijk.size] = n_ijk / n_i
                
        return frequencies
        
      
        
        

    def _calculate_gaussian(self, x, mean, var):
        """ Calculate the normal distribution of the data x given mean and var
        """
        eps = 1e-4  # Added in denominator to prevent division by zero
        #gaussian = 1 / (np.sqrt(2 * np.pi * var) + eps) * np.exp(-(x - mean) ** 2 / (2 * var + eps))
        gaussian = 1 / np.sqrt(2 * np.pi * var + eps) * np.exp(-1 * (x - mean) ** 2 / (2 * var + eps))
        return gaussian

    def _calculate_posteriors(self, X_test):
        """
        Calculate the porterior of each class for the test vector X_test.

        Reminder:
        classification using Bayes Rule P(Y|X) = P(X|Y)*P(Y)/P(X),
                                  or Posterior = Likelihood * Prior / Scaling Factor

        P(Y|X) - The posterior is the probability that sample x is of class y given the
                 feature values of x being distributed according to distribution of y and the prior.
        P(X|Y) - Likelihood of data X given class distribution Y.
                 Continuous data assuming Gaussian distr. -> given by _calculate_gaussian
                 Continuous data without assuming Gaussian distr. -> To be done in a future (obligatory) TP
                 Discrete data  -> given by train (self.frequencies)
        P(Y)   - Prior -> given by train (self.prior)
        P(X)   - Scales the posterior to make it a proper probability distribution.
                 This term is ignored in this implementation since it doesn't affect
                 which class distribution the sample is most likely to belong to.


        Parameters
        ----------
        X_test : array-like, shape = [num_samples, num_features]

        Returns
        -------
        posteriors : array-like, shape = [num_samples, num_classes]
            Returns the probability of the samples for each class in
            the model. The columns correspond to the classes in sorted
            order, as they appear in self.classes.
        """
        """
        1. Calculer les posteriors pour chaque classe pour un ensemble de tests donné 'X_test' et extraire les variables continues et categorical (si elles existents) de 'X_test'
        
        2. Calcul des posteriors pour chaque classe en multipliant les vraisemblances (likelihood) et les priors de chaque classe (séparement entre continues et categoriques)
        """
        X_test = np.array(X_test)
        #print(X_test)
        #print(X_test.shape)
        #print("Shape self continuous : ", self.continuous_features)
        #print("Shape self categorical : ", self.categorical_features)
        #print(self.continuous_features.size)
        
        if self.continuous_features.size != 0:
            
            ######################################################################
            # ToDo: compute P(X|Y) for continuous attributes                     #
            #                                                                    #
            # 1. For every class and feature, for each sample from the samples   #
            #   compute its likelihood                                           #
            #   (num_classes, num_samples, num_features)                         #
            # 2. For every class and sample compute the product prod(P(xi | Y))  #
            #    for the continuous features                                     #
            #    Naive assumption (independence):                                #
            #    P(x1,x2,x3|Y) = P(x1|Y)*P(x2|Y)*P(x3|Y)                         #
            #   (num_samples, num_classes)                                       #
            ######################################################################
            # cont_prod =
            cont_prod = np.zeros((X_test.shape[0], self.classes.size))
            for c in range(self.classes.size):
                # likelihood of X given class Y
                likelihood = self._calculate_gaussian(X_test[:, self.continuous_features],
                                                      self.mean[c],
                                                      self.var[c])
                # product of P(xi | Y)
                prod = np.prod(likelihood, axis=1)
                cont_prod[:, c] = prod
            

        #print(self.categorical_features.size)
        if self.categorical_features.size != 0:
            X = X_test[:, self.categorical_features].astype(int)
            #print("shape X (categorical)... :", X.shape)
            ######################################################################
            # ToDo: compute P(X|Y) for categorical attributes                    #
            #                                                                    #
            # For every class and sample  compute the product prod(P(xi | Y))    #
            #   for the categorical features                                     #
            #   Naive assumption (independence):                                 #
            #   P(x1,x2,x3|Y) = P(x1|Y)*P(x2|Y)*P(x3|Y)                          #
            #   (num_samples, num_classes)                                       #
            ######################################################################
            cat_prod = np.zeros((X.shape[0], self.classes.size))
            #print("shape cat_prod : ", cat_prod.shape)
            #print("shape frequencies : ", self.frequencies.shape)
            for c in range(self.classes.size):
                P_xy = np.zeros((X.shape[0], len(self.categorical_features)))
                for i, f in enumerate(self.categorical_features):
                    #print(c,f,i)
                    #Petit "trick" pour que l'index utilisée par l'attribut categorical soit égal à la valeur de l'index dans la
                    #matrice "frequencies"
                    f = f - self.continuous_features.size
                    P_xy[:, i] = self.frequencies[c, f, X[:, i]]
                cat_prod[:, c] = np.prod(P_xy, axis=1)
            
        

        ######################################################################
        # ToDo: compute P(Y|X)                                               #
        #                                                                    #
        # For every class and sample  compute P(Y|X) = P(X|Y)*P(Y)           #
        #   Naive assumption (independence):                                 #
        #   P(x1,x2,x3|Y) = P(x1|Y)*P(x2|Y)*P(x3|Y)                          #
        #   (num_samples, num_classes)                                       #
        ######################################################################

        if self.continuous_features.size != 0 and self.categorical_features.size != 0:
            """
            Si des attributs continue et categorical sont présents : il faut calculer la probabilité posterior pour chaque class du vecteur 
            test
            Utilisation de Naive Bayes assumption : les "features" sont indpendant de la classe -> utilisation du produit des probabilités 
            conditionnelles de chaque "feature"
            
            """
            posteriors = (cat_prod * cont_prod) * self.priors
        elif self.continuous_features.size != 0:
            posteriors = cont_prod * self.priors
        elif self.categorical_features.size != 0:
            posteriors = cat_prod * self.priors

        
        return posteriors

    def predict(self, X_test):
        """
        Perform classification.
        Classifies each sample as the class that results in the largest P(Y|X) (posterior)

        Parameters
        ----------
        X_test : array-like, shape = [num_samples, n_features]

        Returns
        -------
        predictions : array, shape = [num_samples]
            Predicted target values for X_test
        """
        

        ##########################
        # use only numpy library #
        ##########################

        # predictions =
        posteriors = self._calculate_posteriors(X_test)
        predictions = self.classes[np.argmax(posteriors, axis=1)]
        return predictions





