import numpy as np
import matplotlib.pyplot as plt
from load_mnist import load_mnist

import time

class Layer:
    def __init__(self, input_features, output_features, activation):
        # create vectors / arrays to store Weights and Biases for
        # current layer
        self.W = np.float64(np.random.uniform(-0.0001, 0.0001, (input_features, output_features)))
        #np.random.rand(input_features, output_features)
        self.b = np.random.rand(output_features)
        self.act = activation
        # store the reference to the activation function for this
        # layer
        self.activation = {
            "relu": self.relu,
            "sigmoid": self.sigmoid,
            "softmax": self.softmax,
            "": None
        }[activation]
        # corresponding gradient / backward function
        self.activations_grad = {
            "relu": self.relu_backward,
            "sigmoid": self.sigmoid_backward,
            "softmax": self.softmax_backward,
            "": None
        }[activation]
    
    def softmax(self, z):
        # given the value z let's find exp(z)/sum(exp(z))
        exp_z = np.exp(z)    
        sum_zl = np.sum(exp_z, axis=1)
        return np.divide(exp_z, sum_zl[:, None])
    
    def relu(self, Z):
        # given the value z apply the relu function to it
        return np.maximum(0, Z)
    
    def sigmoid(self, Z):
        # given the value z apply the sigmoid function to it
        return 1.0 / (1.0 + np.exp(-Z))
    
    def sigmoid_backward(self, dA):
        dZ = dA * self.A * (1 - self.A)
        # compute and store gradients for current
        # backward pass run        
        self.dW = np.dot(self.inputs.T, dZ)
        self.db = np.sum(dZ, axis=0, keepdims=False)
        self.dA = np.dot(dZ, self.W.T)
        return self.dA
    
    def relu_backward(self, dA):
        dZ = np.array(dA, copy=True)
        dZ[self.Z <= 0] = 0
        # compute and store gradients for current
        # backward pass run        
        self.dW = np.dot(self.inputs.T, dZ)
        self.db = np.sum(dZ, axis=0, keepdims=False)
        self.dA = np.dot(dZ, self.W.T)
        return self.dA

    def softmax_backward(self, dA):
        self.dW = np.dot(self.inputs.T, dA)
        self.db = np.sum(dA, axis=0, keepdims=False)
        self.dA = np.dot(dA, self.W.T)
        return self.dA
    
    def layer_forward(self, inputs):
        self.inputs = inputs
        self.Z = (inputs @ self.W) + self.b
        if self.activation != None:
            self.A = self.activation(self.Z)
            return self.A
        return self.Z
    
    def layer_backward(self, dA):
        if self.activations_grad != None:
            self.dA = self.activations_grad(dA)
            return self.dA
        return dA


class NeuralNetwork:

    def __init__(self, ):
        # create a place holder list to keep track
        # of all the layer added to the neural network
        self.layers = []

    def add_layer(self, input_features, output_features, activation="relu"):
        # append new layer to the neural network
        self.layers.append(Layer(input_features, output_features, activation))
    
    def random_minibatches(self, n_samples, minibatch_size):
        # we create minibatches of equal-size
        # in order to do so we drop some samples
        # so that the number of samples left are
        # equally divisble by the mini-batch size
        index_array = np.arange(n_samples)
        np.random.shuffle(index_array)
        tk_samples = n_samples - (n_samples % minibatch_size)
        minibatches = np.reshape(index_array[:tk_samples], newshape=[tk_samples//minibatch_size, minibatch_size])
        return minibatches
        
    def predict(self, sample):
        prediction = np.argmax(self.model_forward(sample), axis=1)
        return prediction
    
    def model_forward(self, x_data):
        # compute full forward pass through the neural network
        # firt pass the input through input layer
        layer_out = x_data
        # pass through all layers and also apply respective
        # activation functions
        for layer in self.layers:
            layer_out = layer.layer_forward(layer_out)
        # return the results of the forward pass
        return layer_out
    
    def model_backward(self, X, predicted_y, actual_y):
        # compute gradients for entire neural network
        # using backpropagation technique
        n_samples = len(actual_y)
        dA_prev = predicted_y.copy()
        dA_prev[range(n_samples), np.argmax(actual_y, axis=1)] -= 1
        dA_prev /= n_samples
        # compute gradients for all layer
        for i in reversed(range(len(self.layers))):
            dA_curr = dA_prev
            dA_prev = self.layers[i].layer_backward(dA_curr)
    
    def update_parameters(self, lr = 0.01):
        # apply gradient based update
        for layer in self.layers:
            layer.W -= lr * layer.dW
            layer.b -= lr * layer.db.T
    
    def compute_accuracy(self, predicted_y, actual_y):
        # compute the average accuracy of the model
        return np.mean(np.argmax(predicted_y, axis=1)==np.argmax(actual_y, axis=1))
    
    def compute_loss(self, predicted_y, actual_y):
        # compute overall loss of the model
        n_samples = actual_y.shape[0]
        logprobs = predicted_y[range(n_samples), np.argmax(actual_y, axis=1)]
        loss = np.sum(logprobs) / n_samples
        return loss
    
    def train_model(self, x_train, y_train, x_test, y_test, step_size, epochs, minibatch_size):
        
        # first check what number of samples we are working with
        n_samples = x_train.shape[0]
        
        # variable to store history of cost
        loss_hist = []
        tr_acc_hist = []
        ts_acc_hist = []
        
        # run the training process for prescribed number of training rounds
        for e in range(epochs):
            print(f"Running epoch {e+1} of {epochs}")
            for batch in self.random_minibatches(n_samples=n_samples, minibatch_size=minibatch_size):
                # compute the forward pass
                y_hat = self.model_forward(x_train[batch])
                
                # perform backward pass
                self.model_backward(x_train[batch], y_hat, y_train[batch])
                
                # perform the weight update
                self.update_parameters(lr=step_size)
                
                # compute the loss for this round and store it in
                # history variable so we can use it to draw graphs later
                sample_loss = -np.sum(y_train[batch] * np.log(y_hat), axis=1, keepdims=True)
                current_loss = np.mean(sample_loss)
                loss_hist.append(current_loss)
            
            tr_acc_hist.append(self.evaluate_model(x_train, y_train))
            ts_acc_hist.append(self.evaluate_model(x_test, y_test))
            
        # return history of the cost over all iterations    
        return loss_hist, tr_acc_hist, ts_acc_hist
    
    def evaluate_model(self, x_data, y_data):
        return np.count_nonzero(np.argmax(y_data, axis=1) == self.predict(x_data)) / x_data.shape[0]

    def __repr__(self):
        return self.__str__()
    
    def __str__(self):
        nn_summary = "\n"
        nn_summary += "===================================\n"
        nn_summary += "Network Summary:\n"
        for index, layer in enumerate(self.layers):
            nn_summary += f"\t(Layer_{index+1}): Input={layer.W.shape[0]}, Output={layer.W.shape[1]}, Activation={layer.act}\n"
        nn_summary += "==================================="
        return nn_summary

def loading_data():
    # use provided function to load the data
	return load_mnist()
    
def neural_network():
    # load the auto.csv data by using the given load function
    x_train, y_train, x_test, y_test = loading_data()
    print("Completed Data Loading!")
    
    # train model 1 with only horse power as input feature
    model = NeuralNetwork()
    model.add_layer(784, 20, "sigmoid")
    model.add_layer(20, 20, "relu")
    model.add_layer(20, 10, "softmax")
    
    # print model summary
    print(model)
    
    # train the mode
    begin_time = time.time()
    loss_hist, tr_acc_hist, ts_acc_hist = model.train_model(x_train, y_train, x_test, y_test, step_size=0.5, epochs=25, minibatch_size=64)
    
    # print final summary
    print(f"Training model the on cpu took {time.time() - begin_time} seconds.")
    print(f"Final Accuracy: Train = {tr_acc_hist[-1]} and Test = {ts_acc_hist[-1]}")

    # plot cost over iteration plots
    plt.figure()
    plt.xlabel("Iteration K")
    plt.ylabel("Average Loss")
    plt.plot(loss_hist, linewidth=3)
    plt.savefig("outputs/numpy_nn_loss.pdf", format="pdf", bbox_inches="tight")
    
    # plot training and testing accuracy
    plt.figure()
    plt.xlabel("Epoch K")
    plt.ylabel("Accuracy")
    plt.plot(tr_acc_hist, linewidth=3, label="Train Accuracy")
    plt.plot(ts_acc_hist, linewidth=3, label="Test Accuracy")
    plt.legend(loc="center right")
    plt.savefig("outputs/numpy_nn_accu.pdf", format="pdf", bbox_inches="tight") 
    
if __name__=="__main__":
    neural_network()