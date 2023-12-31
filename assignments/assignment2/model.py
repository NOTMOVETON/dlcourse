import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax, cross_entropy


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.Layer1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.ReLU = ReLULayer()
        self.Layer2 = FullyConnectedLayer(hidden_layer_size, n_output)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        W1 = self.params()['w1']
        W2 = self.params()['w2']
        B1 = self.params()['b1']
        B2 = self.params()['b2']
        W1.grad = np.zeros_like(W1.value)
        W2.grad = np.zeros_like(W1.value)
        B1.grad = np.zeros_like(W1.value)
        B2.grad = np.zeros_like(W1.value)

        # FORWARD CYCLE
        forward1 = self.Layer1.forward(X)
        forward2 = self.ReLU.forward(forward1)
        forward3 = self.Layer2.forward(forward2)

        # SOFTMAX WITH CROSS ENTROPY
        loss, grad = softmax_with_cross_entropy(forward3, y)
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        backward3 = self.Layer2.backward(grad)
        backward2 = self.ReLU.backward(backward3)
        backward1 = self.Layer1.backward(backward2)
        
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        lw1, gw1 = l2_regularization(W1.value, self.reg)
        lb1, gb1 = l2_regularization(B1.value, self.reg)
        lw2, gw2 = l2_regularization(W2.value, self.reg)
        lb2, gb2 = l2_regularization(B2.value, self.reg)

        
        W1.grad += gw1
        W2.grad += gw2
        B1.grad += gb1
        B2.grad += gb2
        
        loss = loss + lw1 + lw2 + lb1 + lb2
        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)


        W1 = self.params()['w1']
        W2 = self.params()['w2']
        B1 = self.params()['b1']
        B2 = self.params()['b2']

        # FORWARD CYCLE
        forward1 = self.Layer1.forward(X)
        forward2 = self.ReLU.forward(forward1)
        forward3 = self.Layer2.forward(forward2)
        
        probs = softmax(forward3)
        pred = np.argmax(probs, axis=1)
        
        return pred

    def params(self):
        result = {
            'w1': self.Layer1.W, 
            'w2': self.Layer2.W,
            'b1': self.Layer1.B, 
            'b2': self.Layer2.B
                 }
        return result
