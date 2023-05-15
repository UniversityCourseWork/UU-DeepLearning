import numpy as np
import matplotlib.pyplot as plt
from load_mnist import load_mnist


class Softmax_Classifier:

    def __init__(self, n_features, n_classes):
        self.n_features = n_features
        self.n_classes = n_classes
        self.W = np.zeros([self.n_features, self.n_classes], np.float64)
        self.b = np.zeros([self.n_classes], np.float64)
    
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
        # compute forward pass for softmax regression
        z = x_data @ self.W + self.b
        return self.softmax(z)
    
    def model_backward(self, predicted_y, actual_y):
        # compute gradients from softmax regression
        # and update the parameters
        pass
    
    def softmax(self, z):
        # given the value z let's find exp(z)/sum(exp(z))
        exp_z = np.exp(z)    
        sum_zl = np.sum(exp_z, axis=1)
        return np.divide(exp_z, sum_zl[:, None])
    
    
    def train_model(self, x_train, y_train, x_test, y_test, step_size, epochs, minibatch_size):
        
        # first check what number of samples we are working with
        n_samples = x_train.shape[0]
        
        # variable to store history of cost
        loss_hist = []
        tr_acc_hist = []
        ts_acc_hist = []
        
        # run the training process for prescribed number of training rounds
        for i in range(0, epochs):
            for batch in self.random_minibatches(n_samples=n_samples, minibatch_size=minibatch_size):
                # compute the forward pass
                y_hat = self.model_forward(x_train[batch])
                
                cost_J = y_hat - y_train[batch]
                
                # compute gradients
                partial_JW = (x_train[batch].T @ cost_J) / n_samples
                partial_Jb = np.sum(cost_J, axis=0) / n_samples
                
                # update weights and bias terms
                self.W -= step_size * partial_JW
                self.b -= step_size * partial_Jb
                
            # compute the loss for this round and store it in
            # history variable so we can use it to draw graphs later
            sample_loss = -np.sum(y_train[batch] * np.log(y_hat), axis=1, keepdims=True)
            current_loss = np.mean(sample_loss)
            loss_hist.append(current_loss)
            tr_acc_hist.append(self.evaluate_model(x_train[batch], y_train[batch]))
            ts_acc_hist.append(self.evaluate_model(x_test, y_test))
            
        # return history of the cost over all iterations    
        return loss_hist, tr_acc_hist, ts_acc_hist

    def evaluate_model(self, x_data, y_data):
        return np.count_nonzero(np.argmax(y_data, axis=1) == self.predict(x_data)) / x_data.shape[0]


def loading_data():
    # use provided function to load the data
	return load_mnist()


def softmax_classification():
    # load the auto.csv data by using the given load function
    x_train, y_train, x_test, y_test = loading_data()
    
    print("Completed Data Loading! Training Model...")
    
    # train model 1 with only horse power as input feature
    model = Softmax_Classifier(784, 10)
    loss_hist, tr_acc_hist, ts_acc_hist = model.train_model(x_train, y_train, x_test, y_test, step_size=0.5, epochs=25, minibatch_size=64)
    
    # plot cost over iteration plots
    plt.figure()
    plt.xlabel("Epochs K")
    plt.ylabel("Average Loss")
    plt.plot(loss_hist, linewidth=3)
    plt.savefig("outputs/minibatch_softmax_loss.pdf", format="pdf", bbox_inches="tight")
    
    # plot training and testing accuracy
    plt.figure()
    plt.xlabel("Iteration K")
    plt.ylabel("Accuracy")
    plt.plot(tr_acc_hist, linewidth=3, label="Train Accuracy")
    plt.plot(ts_acc_hist, linewidth=3, label="Test Accuracy")
    plt.legend(loc="center right")
    plt.savefig("outputs/minibatch_softmax_accu.pdf", format="pdf", bbox_inches="tight")    
    
    # save weight vectors as 28x28 image plots
    for i, weights in enumerate(model.W.T):
        plt.figure()
        plt.imshow(weights.reshape(28, 28))
        plt.savefig(f"weight_{i}.png")
    
    
if __name__=="__main__":
    softmax_classification()