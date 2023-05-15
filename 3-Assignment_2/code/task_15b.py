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

class CNN_SMALL(nn.Module):

    def __init__(self):
        super(CNN_SMALL, self).__init__()
        # convolutional layers
        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(25088, 10),
        )
    
    def forward(self, x):
        # apply convolutional layers
        x = self.convolutional_layers(x)
        # flatten the input layer
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        # apply the linear layers as well as the softmax function
        x = self.linear_layers(x)
        x = F.softmax(x, dim=1)
        return x


class CNN_MEDIUM(nn.Module):

    def __init__(self):
        super(CNN_MEDIUM, self).__init__()
        # convolutional layers
        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding="same"),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding="same"),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding="same"),
            nn.LeakyReLU(),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(25088, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        # apply convolutional layers
        x = self.convolutional_layers(x)
        # flatten the input layer
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        # apply the linear layers as well as the softmax function
        x = self.linear_layers(x)
        x = F.softmax(x, dim=1)
        return x


class CNN_LARGE(nn.Module):

    def __init__(self):
        super(CNN_LARGE, self).__init__()
        # convolutional layers
        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding="same"),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding="same"),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding="same"),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding="same"),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding="same"),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding="same"),
            nn.LeakyReLU(),
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(50176, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 10)
        )
        
    def forward(self, x):
        # apply convolutional layers
        x = self.convolutional_layers(x)
        # flatten the input layer
        x = torch.flatten(x, 1) # flatten all dimensions except the batch dimension
        # apply the linear layers as well as the softmax function
        x = self.linear_layers(x)
        x = F.softmax(x, dim=1)
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
    optimizer = optim.Adam(model.parameters(), lr=step_size)

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
    modelSMALL = CNN_SMALL()
    modelSMALL.to(device=device)
    modelMEDIUM = CNN_MEDIUM()
    modelMEDIUM.to(device=device)
    modelLARGE = CNN_LARGE()
    modelLARGE.to(device=device)
    
    # print model summary
    print(f"Training model {modelSMALL} on {device} with {sum(p.numel() for p in modelSMALL.parameters() if p.requires_grad)} trainable weights.")
    begin_time = time.time()
    # train the model
    loss_hist_small, tr_acc_hist_small, ts_acc_hist_small = train_mnist(model=modelSMALL, trainset=trainset, testset=testset, step_size=1e-3, epochs=25, minibatch_size=64)
    # print final summary
    print(f"Training model the on cpu took {time.time() - begin_time} seconds.")
    print(f"Final Accuracy: Train = {tr_acc_hist_small[-1]} and Test = {ts_acc_hist_small[-1]}")

    # print model summary
    print(f"Training model {modelMEDIUM} on {device} with {sum(p.numel() for p in modelMEDIUM.parameters() if p.requires_grad)} trainable weights.")
    begin_time = time.time()
    # train the model
    loss_hist_medium, tr_acc_hist_medium, ts_acc_hist_medium = train_mnist(model=modelMEDIUM, trainset=trainset, testset=testset, step_size=5e-4, epochs=25, minibatch_size=64)
    # print final summary
    print(f"Training model the on cpu took {time.time() - begin_time} seconds.")
    print(f"Final Accuracy: Train = {tr_acc_hist_medium[-1]} and Test = {ts_acc_hist_medium[-1]}")

    # print model summary
    print(f"Training model {modelLARGE} on {device} with {sum(p.numel() for p in modelLARGE.parameters() if p.requires_grad)} trainable weights.")
    begin_time = time.time()
    # train the model
    loss_hist_large, tr_acc_hist_large, ts_acc_hist_large = train_mnist(model=modelLARGE, trainset=trainset, testset=testset, step_size=1e-4, epochs=25, minibatch_size=64)
    # print final summary
    print(f"Training model the on cpu took {time.time() - begin_time} seconds.")
    print(f"Final Accuracy: Train = {tr_acc_hist_large[-1]} and Test = {tr_acc_hist_large[-1]}")

    # plot cost over iteration plots
    plt.figure()
    plt.xlabel("Iteration K")
    plt.ylabel("Average Loss")
    plt.plot(loss_hist_small, label="Small CNN Model")
    plt.plot(loss_hist_medium, label="Medium CNN Model")
    plt.plot(loss_hist_large, label="Large CNN Model")
    plt.legend(loc="upper right")
    plt.savefig("outputs/task15b_model_loss.pdf", format="pdf", bbox_inches="tight")
    
    # plot training and testing accuracy
    plt.figure()
    plt.xlabel("Epoch K")
    plt.ylabel("Accuracy")
    plt.plot(ts_acc_hist_small, linewidth=3, label="Small CNN Model")
    plt.plot(ts_acc_hist_medium, linewidth=3, label="Medium CNN Model")
    plt.plot(ts_acc_hist_large, linewidth=3, label="Large CNN Model")
    plt.legend(loc="lower right")
    plt.savefig("outputs/task15b_model_accu.pdf", format="pdf", bbox_inches="tight") 
    