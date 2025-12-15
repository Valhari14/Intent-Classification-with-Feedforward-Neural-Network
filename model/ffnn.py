import numpy as np
import numpy.typing as npt

from model.model_utils import softmax, relu, relu_prime
from typing import Tuple


class NeuralNetwork(object):
    def __init__(
        self, 
        input_size: int,
        hidden_size: int, 
        num_classes: int,
        seed: int = 1
    ):
        """
        Initialize neural network's weights and biases.
        Args:
            input_size: dimension of input features (V)
            hidden_size: number of neurons in hidden layer (150)
            num_classes: number of output classes (K)
            seed: random seed for reproducibility
        """
        ############################# STUDENT SOLUTION ####################
        # YOUR CODE HERE
        #     TODO:
        #         1) Set a seed so that your model is reproducible
        #         2) Initialize weight matrices and biases with uniform
        #         distribution in the range (-1, 1).
        
        np.random.seed(seed) # Set seed for reproducibility

        # Initialize weights and biases for hidden layer
        # W1 shape: (hidden_size, input_size)
        # b1 shape: (hidden_size, 1)
        self.w1 = np.random.uniform(-1,1,(hidden_size,input_size)) 
        self.b1 =  np.random.uniform(-1,1,(hidden_size,1)) 
       
        # Initialize weights and biases for output layer
        # W2 shape: (num_classes, hidden_size)
        # b2 shape: (num_classes, 1)
        self.w2 = np.random.uniform(-1,1,(num_classes,hidden_size)) 
        self.b2 =  np.random.uniform(-1,1,(num_classes,1))
        ###################################################################

    def forward(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """
        Forward pass with X as input matrix, returning the model prediction
        Y_hat.
        Args:
            X: Input matrix of shape (input_size, M) where M is batch size
        Returns:
            Y_hat: Output predictions of shape (num_classes, M)
        """
        ######################### STUDENT SOLUTION #########################
        # YOUR CODE HERE
        #     TODO:
        #         1) Perform only a forward pass with X as input.
        
        # First layer forward pass: Input to hidden layer then apply ReLU activation
        # z1 shape: (hidden_size, M)
        # a1 shape: (hidden_size, M)
        self.z1 = np.dot(self.w1, X) + self.b1
        self.a1 = relu(self.z1)

        # Second layer forward pass: Hidden layer to output layer then apply Softmax activation
        #z2 shape: (num_classes, M)
        # Y_hat shape: (num_classes, M)
        z2 = np.dot(self.w2, self.a1) + self.b2
        Y_hat = softmax(z2)

        return Y_hat
        #####################################################################

    def predict(self, X: npt.ArrayLike) -> npt.ArrayLike:
        """
        Create a prediction matrix with `self.forward()`
        """
        ######################### STUDENT SOLUTION ###########################
        # YOUR CODE HERE
        #     TODO:
        #         1) Create a prediction matrix of the intent data using
        #         `self.forward()` function. The shape of prediction matrix
        #         should be similar to label matrix produced with
        #         `labels_matrix()`

        Y_hat = self.forward(X)
        prediction_idx = np.argmax(Y_hat, axis = 0)  # Get the index of the max probability for each column (examples)
        K = Y_hat.shape[0]  # number of classes
        M = Y_hat.shape[1]  # number of examples
        
        ''' To shape the prediction matrix similar to the labels_matrix(num_classes, num_examples), 
        create a zero matrix and set the predicted class indices to 1 '''
        prediction_matrix = np.zeros((K, M))
        prediction_matrix[prediction_idx, np.arange(M)] = 1 

        return prediction_matrix
        ######################################################################

    def backward(
        self, 
        X: npt.ArrayLike, 
        Y: npt.ArrayLike
    ) -> Tuple[npt.ArrayLike, npt.ArrayLike, npt.ArrayLike, npt.ArrayLike]:
        """
        Backpropagation algorithm.
        """
        ########################## STUDENT SOLUTION ###########################
        # YOUR CODE HERE
        #     TODO:
        #         1) Perform forward pass, then backpropagation
        #         to get gradient for weight matrices and biases
        #         2) Return the gradient for weight matrices and biases

        Y_hat = self.forward(X) # Forward pass to get latest predictions
        M = X.shape[1]

        # First Backward pass from output layer to hidden layer
        '''
        Loss for output layer = (Y_hat - Y) -> shape: (K, M)
        Gradient for W2 =  1/M *(loss @ a1^T) -> shape: (K, hidden_size)
        Gradient for b2 =  1/M * sum of loss across all examples -> shape: (K, 1)
        '''

        delta2 = Y_hat - Y
        dw2 = np.dot(delta2, self.a1.T)   
        db2 = np.sum(delta2, axis = 1, keepdims = True) 

        # Second Backward pass from hidden layer to input layer
        '''
        Loss for hidden layer =  (W2^T * delta2) * ReLU'(z1) -> shape: (hidden_size, M)
        Gradient for W1 =  1/M * (loss @ X^T) -> shape: (hidden_size, input_size)
        Gradient for b1 =  1/M * sum of loss across all examples -> shape: (K, 1)
        '''
        delta1 = np.dot(self.w2.T, delta2) * relu_prime(self.z1)
        dw1 = np.dot(delta1, X.T)
        db1 = np.sum(delta1, axis = 1, keepdims = True) 

        return dw1, db1, dw2, db2
        #######################################################################


def compute_loss(pred: npt.ArrayLike, truth: npt.ArrayLike) -> float:
    """
    Compute the cross entropy loss.

    Args:
        pred (Y_hat): Predicted probabilities (K x M)
        truth (Y): One-hot encoded ground truth (K x M)
    
    Returns:
        Average cross entropy loss
    """
    ########################## STUDENT SOLUTION ###########################
    # YOUR CODE HERE
    #     TODO:
    #         1) Compute the cross entropy loss between your model prediction
    #         and the ground truth.
    # Cross-entropy loss for batch of M examples =  - (1/M) * sum of (Y * log(Y_hat))

    M = truth.shape[1] # number of examples
    epsilon = 1e-15 # small constant to avoid log(0)
    pred = np.clip(pred, epsilon, 1 - epsilon)  # ensures no 0 or 1
    log_pred = np.log(pred)
    loss = -np.sum(truth * log_pred) / M

    return loss
    #######################################################################