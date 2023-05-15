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

class CNN_RELU_B(nn.Module):

    def __init__(self):
        super(CNN_RELU_B, self).__init__()
        # convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        # Computing Network Shape
        # Input:                            batch_size x  1 x 28 x 28 
        # CovL1: [(28 - 3 + 2*1 / 1) + 1] = batch_size x  8 x 28 x 28
        # ReLU:                             batch_size x  8 x 28 x 28
        # MaxPool:                          batch_size x  8 x 14 x 14
        # CovL2: [(14 - 3 + 2*1 / 1) + 1] = batch_size x 16 x 14 x 14
        # ReLU:                             batch_size x 16 x 14 x 14
        # MaxPool:                          batch_size x 16 x  7 x  7
        # CovL3: [( 7 - 3 + 2*1 / 1) + 1] = batch_size x 32 x  7 x  7
        # ReLU:                             batch_size x 32 x  7 x  7
        # FLAT:                             batch_size x 1568
        
        # create a linear list using PyTorch module list
        # self.linears = nn.ModuleList(fully_connected_layers)
        self.linears = nn.ModuleList([nn.Linear(1568, 10)])
        

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=(2,2), stride=2)
        x = F.max_pool2d(F.relu(self.conv2(x)), kernel_size=(2,2), stride=2)
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        for layer in self.linears:
            x = layer(x)
        x = F.softmax(x, dim=0)
        return x

class CNN_RELU_A(nn.Module):

    def __init__(self):
        super(CNN_RELU_A, self).__init__()
        # convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        # Computing Network Shape
        # Input:                            batch_size x  1 x 28 x 28 
        # CovL1: [(28 - 3 + 2*1 / 1) + 1] = batch_size x  8 x 28 x 28
        # ReLU:                             batch_size x  8 x 28 x 28
        # MaxPool:                          batch_size x  8 x 14 x 14
        # CovL2: [(14 - 3 + 2*1 / 1) + 1] = batch_size x 16 x 14 x 14
        # ReLU:                             batch_size x 16 x 14 x 14
        # MaxPool:                          batch_size x 16 x  7 x  7
        # CovL3: [( 7 - 3 + 2*1 / 1) + 1] = batch_size x 32 x  7 x  7
        # ReLU:                             batch_size x 32 x  7 x  7
        # FLAT:                             batch_size x 1568
        
        # create a linear list using PyTorch module list
        # self.linears = nn.ModuleList(fully_connected_layers)
        self.linears = nn.ModuleList([nn.Linear(1568, 10)])
        

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=(2,2), stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=(2,2), stride=2))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        for layer in self.linears:
            x = layer(x)
        x = F.softmax(x, dim=0)
        return x


class CNN_TANH_B(nn.Module):

    def __init__(self):
        super(CNN_TANH_B, self).__init__()
        # convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        # Computing Network Shape
        # Input:                            batch_size x  1 x 28 x 28 
        # CovL1: [(28 - 3 + 2*1 / 1) + 1] = batch_size x  8 x 28 x 28
        # tanh:                             batch_size x  8 x 28 x 28
        # MaxPool:                          batch_size x  8 x 14 x 14
        # CovL2: [(14 - 3 + 2*1 / 1) + 1] = batch_size x 16 x 14 x 14
        # tanh:                             batch_size x 16 x 14 x 14
        # MaxPool:                          batch_size x 16 x  7 x  7
        # CovL3: [( 7 - 3 + 2*1 / 1) + 1] = batch_size x 32 x  7 x  7
        # tanh:                             batch_size x 32 x  7 x  7
        # FLAT:                             batch_size x 1568
        
        # create a linear list using PyTorch module list
        # self.linears = nn.ModuleList(fully_connected_layers)
        self.linears = nn.ModuleList([nn.Linear(1568, 10)])
        

    def forward(self, x):
        x = F.max_pool2d(F.tanh(self.conv1(x)), kernel_size=(2,2), stride=2)
        x = F.max_pool2d(F.tanh(self.conv2(x)), kernel_size=(2,2), stride=2)
        x = F.tanh(self.conv3(x))
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        for layer in self.linears:
            x = layer(x)
        x = F.softmax(x, dim=0)
        return x

class CNN_TANH_A(nn.Module):

    def __init__(self):
        super(CNN_TANH_A, self).__init__()
        # convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        
        # Computing Network Shape
        # Input:                            batch_size x  1 x 28 x 28 
        # CovL1: [(28 - 3 + 2*1 / 1) + 1] = batch_size x  8 x 28 x 28
        # tanh:                             batch_size x  8 x 28 x 28
        # MaxPool:                          batch_size x  8 x 14 x 14
        # CovL2: [(14 - 3 + 2*1 / 1) + 1] = batch_size x 16 x 14 x 14
        # tanh:                             batch_size x 16 x 14 x 14
        # MaxPool:                          batch_size x 16 x  7 x  7
        # CovL3: [( 7 - 3 + 2*1 / 1) + 1] = batch_size x 32 x  7 x  7
        # tanh:                             batch_size x 32 x  7 x  7
        # FLAT:                             batch_size x 1568
        
        # create a linear list using PyTorch module list
        # self.linears = nn.ModuleList(fully_connected_layers)
        self.linears = nn.ModuleList([nn.Linear(1568, 10)])
        

    def forward(self, x):
        x = F.tanh(F.max_pool2d(self.conv1(x), kernel_size=(2,2), stride=2))
        x = F.tanh(F.max_pool2d(self.conv2(x), kernel_size=(2,2), stride=2))
        x = F.tanh(self.conv3(x))
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        for layer in self.linears:
            x = layer(x)
        x = F.softmax(x, dim=0)
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
        #print(f"Running epoch {e+1} of {epochs}")
        # put model in train mode
        model.train()
        running_loss = 0
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
            running_loss += loss.item() * y.shape[0]
            # carry out a backward pass to compute gradients
            loss.backward()
            # take a step to update the parameters
            optimizer.step()

        # evaluate model on both train as well as test sets
        loss_hist.append(running_loss / len(trainset))
        tr_acc_hist.append(evaluate(model=model, dt_loader=train_loader)["accuracy"])
        ts_acc_hist.append(evaluate(model=model, dt_loader=test_loader)["accuracy"])

        # print out results
        print(f"Epoch {e+1} of {epochs} : Loss = {'{:.4f}'.format(loss_hist[-1])} | Accuracy: Train = {'{:.4f}'.format(tr_acc_hist[-1])} and Test = {'{:.4f}'.format(ts_acc_hist[-1])}")

    # return trained stats
    return loss_hist, tr_acc_hist, ts_acc_hist

if __name__=="__main__":
    # load datasets / download if not present on disk
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = torchvision.datasets.MNIST("./dataset/", train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST("./dataset/", train=False, transform=transform)

    # create model using our new implementation
    # that uses PyTorch deep learning library
    modelRB = CNN_RELU_B()
    modelRB.to(device=device)
    modelRA = CNN_RELU_A()
    modelRA.to(device=device)
    modelTB = CNN_TANH_B()
    modelTB.to(device=device)
    modelTA = CNN_TANH_A()
    modelTA.to(device=device)
    
    # print model summary
    print(f"Training model {modelRB} on {device} with {sum(p.numel() for p in modelRB.parameters() if p.requires_grad)} trainable weights.")
    begin_time = time.time()
    # train the model
    loss_hist_rb, tr_acc_hist_rb, ts_acc_hist_rb = train_mnist(model=modelRB, trainset=trainset, testset=testset, step_size=0.75, epochs=25, minibatch_size=64)
    # print final summary
    print(f"Training model the on cpu took {time.time() - begin_time} seconds.")
    print(f"Final Accuracy: Train = {tr_acc_hist_rb[-1]} and Test = {ts_acc_hist_rb[-1]}")

    # print model summary
    print(f"Training model {modelRA} on {device} with {sum(p.numel() for p in modelRA.parameters() if p.requires_grad)} trainable weights.")
    begin_time = time.time()
    # train the model
    loss_hist_ra, tr_acc_hist_ra, ts_acc_hist_ra = train_mnist(model=modelRA, trainset=trainset, testset=testset, step_size=0.75, epochs=25, minibatch_size=64)
    # print final summary
    print(f"Training model the on cpu took {time.time() - begin_time} seconds.")
    print(f"Final Accuracy: Train = {tr_acc_hist_ra[-1]} and Test = {ts_acc_hist_ra[-1]}")

    # print model summary
    print(f"Training model {modelTB} on {device} with {sum(p.numel() for p in modelTB.parameters() if p.requires_grad)} trainable weights.")
    begin_time = time.time()
    # train the model
    loss_hist_tb, tr_acc_hist_tb, ts_acc_hist_tb = train_mnist(model=modelTB, trainset=trainset, testset=testset, step_size=0.75, epochs=25, minibatch_size=64)
    # print final summary
    print(f"Training model the on cpu took {time.time() - begin_time} seconds.")
    print(f"Final Accuracy: Train = {tr_acc_hist_tb[-1]} and Test = {ts_acc_hist_tb[-1]}")

    # print model summary
    print(f"Training model {modelTA} on {device} with {sum(p.numel() for p in modelTA.parameters() if p.requires_grad)} trainable weights.")
    begin_time = time.time()
    # train the model
    loss_hist_ta, tr_acc_hist_ta, ts_acc_hist_ta = train_mnist(model=modelTA, trainset=trainset, testset=testset, step_size=0.75, epochs=25, minibatch_size=64)
    # print final summary
    print(f"Training model the on cpu took {time.time() - begin_time} seconds.")
    print(f"Final Accuracy: Train = {tr_acc_hist_ta[-1]} and Test = {ts_acc_hist_ta[-1]}")


    # plot cost over iteration plots
    plt.figure()
    plt.xlabel("Iteration K")
    plt.ylabel("Average Loss")
    plt.plot(loss_hist_rb, label="ReLU Before")
    plt.plot(loss_hist_ra, label="ReLU After")
    plt.plot(loss_hist_tb, label="TanH Before")
    plt.plot(loss_hist_ta, label="TanH After")
    plt.legend(loc="upper right")
    plt.savefig("outputs/task13_loss.pdf", format="pdf", bbox_inches="tight")
    
    # plot training and testing accuracy
    plt.figure()
    plt.xlabel("Epoch K")
    plt.ylabel("Accuracy")
    plt.plot(tr_acc_hist_rb, linewidth=3, label="ReLU Before")
    plt.plot(tr_acc_hist_ra, linewidth=3, label="ReLU After")
    plt.plot(tr_acc_hist_tb, linewidth=3, label="TanH Before")
    plt.plot(tr_acc_hist_ta, linewidth=3, label="TanH After")
    plt.legend(loc="center right")
    plt.savefig("outputs/task13_tr_accu.pdf", format="pdf", bbox_inches="tight") 
    
    plt.figure()
    plt.xlabel("Epoch K")
    plt.ylabel("Accuracy")
    plt.plot(ts_acc_hist_rb, linewidth=3, label="ReLU Before")
    plt.plot(ts_acc_hist_ra, linewidth=3, label="ReLU After")
    plt.plot(ts_acc_hist_tb, linewidth=3, label="TanH Before")
    plt.plot(ts_acc_hist_ta, linewidth=3, label="TanH After")
    plt.legend(loc="center right")
    plt.savefig("outputs/task13_ts_accu.pdf", format="pdf", bbox_inches="tight") 
    