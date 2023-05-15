import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import time

# use GPU if available
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(784, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 10)

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        x = torch.sigmoid(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def evaluate(model, dt_loader):
    """Evaluation function to check performance"""
    assert(dt_loader is not None)
    
    # start evaluation of the model
    model.eval()
    samples, correct = 0, 0
    
    with torch.no_grad():
        for x, y in dt_loader:
            
            x, y = x.to(device), y.to(device)
            
            y_ = model(x)
            _, predicted = torch.max(y_.detach(), 1)
            
            samples += y.shape[0]
            correct += (predicted == y).sum().item()
    
    # return evaluation statistics
    return {"accuracy" : correct/samples}

# a function to run model training
def train_mnist(model, trainset, testset, step_size, epochs, minibatch_size):
    # create dataloaders
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=minibatch_size, shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=minibatch_size, shuffle=True, num_workers=8)
    
    # create an optimizer to train
    optimizer = optim.SGD(model.parameters(), lr=step_size)

    # variable to store history of cost
    loss_hist = []
    tr_acc_hist = []
    ts_acc_hist = []

    # train the neural network mode
    for e in range(epochs):
        print(f"Running epoch {e+1} of {epochs}")
        # put model in train mode
        model.train()
        # run through entire dataset for every epoch
        for x, y in train_loader:
            # stage input to processing device (cpu or gpu)
            x, y = x.to(device), y.to(device)
            # get prediction from the model
            y_hat = model(x)
            # reset optimizer memory for current backward pass
            optimizer.zero_grad()
            # comput the running loss for current minibatch
            loss = nn.CrossEntropyLoss()(y_hat, y)
            running_loss = loss.item() * y.shape[0]
            loss_hist.append(running_loss)
            # carry out a backward pass to compute gradients
            loss.backward()
            # take a step to update the parameters
            optimizer.step()

        # evaluate model on both train as well as test sets
        tr_acc_hist.append(evaluate(model=model, dt_loader=train_loader)["accuracy"])
        ts_acc_hist.append(evaluate(model=model, dt_loader=test_loader)["accuracy"])

    # return trained stats
    return loss_hist, tr_acc_hist, ts_acc_hist

if __name__=="__main__":
    # load datasets / download if not present on disk
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST("./dataset/", train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST("./dataset/", train=False, transform=transform)

    # create model using our new implementation
    # that uses PyTorch deep learning library
    model = NeuralNetwork()
    model.to(device=device)
    
    # print model summary
    print(f"Training model {model} on {device} with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable weights.")
    
    begin_time = time.time()
    # train the model
    loss_hist, tr_acc_hist, ts_acc_hist = train_mnist(model=model, trainset=trainset, testset=testset, step_size=0.75, epochs=25, minibatch_size=64)
    
    # print final summary
    print(f"Training model the on {device} took {time.time() - begin_time} seconds.")
    print(f"Final Accuracy: Train = {tr_acc_hist[-1]} and Test = {ts_acc_hist[-1]}")

    # plot cost over iteration plots
    plt.figure()
    plt.xlabel("Iteration K")
    plt.ylabel("Average Loss")
    plt.plot(loss_hist, linewidth=3)
    plt.savefig("outputs/pytorch_nn_loss.pdf", format="pdf", bbox_inches="tight")
    
    # plot training and testing accuracy
    plt.figure()
    plt.xlabel("Epoch K")
    plt.ylabel("Accuracy")
    plt.plot(tr_acc_hist, linewidth=3, label="Train Accuracy")
    plt.plot(ts_acc_hist, linewidth=3, label="Test Accuracy")
    plt.legend(loc="center right")
    plt.savefig("outputs/pytorch_nn_accu.pdf", format="pdf", bbox_inches="tight") 
    