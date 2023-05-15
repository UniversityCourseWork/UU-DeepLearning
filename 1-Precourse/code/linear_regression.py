#import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from load_auto import load_auto


class Linear_Regression:

    def __init__(self, n_features):
        self.n_features = n_features
        self.weights = np.zeros(self.n_features, np.float64)
        self.bias = 0.0
        
    def predict(self, sample):
        prediction = np.dot(sample, self.weights.T) + self.bias
        return prediction
    
    def train(self, x_train, y_train, step_size, rounds):
        
        # first check what number of samples we are working with
        n_samples = x_train.shape[0]
        
        # reshape y to conform to shapes we used for weights etc.
        y_train = y_train.reshape(-1)
        
        # variable to store history of cost
        J_history = []
        
        # run the training process for prescribed number of training rounds
        for i in range(0, rounds):
            
            # compute the prediction vector z for all samples
            z = np.dot(x_train, self.weights.T) + self.bias
            
            # compute Loss of all training data points
            error = (z - y_train)
            
            # compute cost for this round
            J = (1 / n_samples) * np.sum(error ** 2)
            
            # printout training cost for this round
            J_history.append(J)
            
            # compute the gradient update for weights
            partial_JW = (2 / n_samples) * (error @ x_train)
            partial_Jb = (2 / n_samples) * np.sum(error)
            
            # apply the update rule to update weights and bias
            self.weights = self.weights - (step_size * partial_JW)
            self.bias = self.bias - (step_size * partial_Jb)
    
        # return history of the cost over all iterations    
        return J_history


def loading_data():
    # use provided function to load the data
	return load_auto()


def normalize_data(x_train):
    # compute mean of the training data
    data_mean = np.mean(x_train, 0)
    # compute standard deviation of the data
    data_std = np.std(x_train, 0)
    # normaliza the data using its mean and standard deviation
    normal_data = (x_train - data_mean)  / data_std
    # send back normalized data
    return normal_data, data_mean, data_std
    

def linear_regression():
    # load the auto.csv data by using the given load function
    x_train, y_train = loading_data()
    
    # normalize the training data
    x_train_norm, mean, std = normalize_data(x_train)
    
    # train model 1 with only horse power as input feature
    model1 = Linear_Regression(1)
    m1_cost = model1.train(x_train_norm[:, 2].reshape(392, 1), y_train, 0.50, 10)
    
    # train model 2 with all variables as input features
    model2 = Linear_Regression(7)
    m2_cost = model2.train(x_train_norm, y_train, 0.2, 100)
    
    # arrow properties to be used to mark final costs
    prop = dict(arrowstyle="-|>,head_width=0.4,head_length=0.8",shrinkA=0,shrinkB=0, color="green")
    
    # plot cost over iteration plots
    plt.figure()
    plt.xlabel("Iteration K")
    plt.ylabel("Cost J")
    plt.plot(m1_cost, linewidth=3)
    plt.annotate("Final Cost: {:.2f}".format(m1_cost[-1]), xy=(9,m1_cost[-1]+20), xytext=(6, 220), arrowprops=prop, color="green")
    plt.savefig("plot1.pdf", format="pdf", bbox_inches="tight")
    
    plt.figure()
    plt.xlabel("Iteration K")
    plt.ylabel("Cost J")
    plt.plot(m2_cost, linewidth=3)
    plt.annotate("Final Cost: {:.2f}".format(m2_cost[-1]), xy=(99,m2_cost[-1]+20), xytext=(70, 250), arrowprops=prop, color="green")
    plt.savefig("plot2.pdf", format="pdf", bbox_inches="tight")
    
    # plot the horsepower vs mpg scatter plot with regression line
    sample_data = [x for x in range(50, 225)]
    regression_plot = [model1.predict((x-mean[2])/std[2]) for x in sample_data]
    plt.figure()
    plt.xlabel("Horsepower")
    plt.ylabel("Miles per gallon (mpg)")
    plt.scatter(x_train[:, 2], y_train, color="green", label="Data (horsepower vs mpg)")
    plt.plot(sample_data, regression_plot, linewidth=4, color="red", label="Trained Linear Model")
    plt.legend()
    plt.savefig("plot3.pdf", format="pdf", bbox_inches="tight")
    
    # print final costs to verify results
    print(f"Final Cost for single-variable Linear Regression: {m1_cost[-1]}")
    print(f"Final Cost for multi-variable Linear Regression: {m2_cost[-1]}")
    
if __name__=="__main__":
    linear_regression()