import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.dataset
from torchmetrics.functional import dice

from load_warwick import WARWICKDataset
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import time

from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sn
import pandas as pd



# use GPU if available
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class CNN_SEGMENT(nn.Module):

    def __init__(self):
        super(CNN_SEGMENT, self).__init__()
        # convolutional layers
        self.convolutional_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Dropout(p = 0.4),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Dropout(p = 0.4),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5, stride=1, padding="same"),
            nn.LeakyReLU(),
            nn.Dropout(p = 0.4),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding="same"),
            nn.LeakyReLU(),
            nn.Dropout(p = 0.4),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding="same"),
            nn.LeakyReLU(),
            nn.Dropout(p = 0.4),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding="same"),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1, padding="same", bias=False),
        )
        
    def forward(self, x):
        # apply convolutional layers
        x = self.convolutional_layers(x)
        #x = F.softmax(x, dim=1) # softmax is applied in the loss function, so no need to apply here
        return x

def evaluate(model, dt_loader):
    """Evaluation function to check performance"""
    assert(dt_loader is not None)
    
    # start evaluation of the model
    model.eval()
    samples, scores = 0, 0

    with torch.no_grad():
        for x, _, y in dt_loader:
            x, y = x.to(device), y.to(device)
            
            y_ = model(x)
            _, predicted = torch.max(y_.detach(), 1)
            dice_score = [dice(predicted[i].int(), y[i].int()) for i in range(len(predicted))]
            
            samples += y.shape[0]
            scores += sum(dice_score)
    
    # return the average dice score
    return {"accuracy" : scores/samples}

# a function to run model training
def train_warwick(model, trainset, testset, step_size, epochs, minibatch_size):
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
        for x, y, y_or in train_loader:
            # stage input to processing device (cpu or gpu)
            x, y = x.to(device), y.to(device)
            # get prediction from the model
            y_hat = model(x)
            #_, y_hat = torch.max(y_predict.detach(), dim=1)
            # reset optimizer memory for current backward pass
            optimizer.zero_grad()
            # comput the running loss for current minibatch
            loss = nn.CrossEntropyLoss(reduction='mean')(y_hat, y)
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
    trainset = WARWICKDataset(root_dir="WARWICK", train=True, transform=transform, target_transform=transform, run_device=device)
    testset = WARWICKDataset(root_dir="WARWICK", train=False, transform=transform, target_transform=transform, run_device=device)
    
    # create model using our new implementation
    # that uses PyTorch deep learning library
    modelSEGMENT = CNN_SEGMENT()
    modelSEGMENT.to(device=device)
    
    # print model summary
    print(f"Training model {modelSEGMENT} on {device} with {sum(p.numel() for p in modelSEGMENT.parameters() if p.requires_grad)} trainable weights.")
    begin_time = time.time()
    # train the model
    loss_hist_seg, tr_acc_hist_seg, ts_acc_hist_seg= train_warwick(model=modelSEGMENT, trainset=trainset, testset=testset, step_size=1e-3, epochs=100, minibatch_size=10)
    # print final summary
    print(f"Training model the on cpu took {time.time() - begin_time} seconds.")
    print(f"Final Accuracy: Train = {tr_acc_hist_seg[-1]} and Test = {ts_acc_hist_seg[-1]}")

    # plot cost over iteration plots
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=8)
    modelSEGMENT.eval()
    with torch.no_grad():
        n_rows, n_cols = 6, 2
        f, axarr = plt.subplots(n_rows, n_cols, figsize=(10, 30))
        row = 0
        for data, _, label in testloader:
            data, label = data.to(device), label.to(device)
            
            # make prediction for output mask
            y_predict = modelSEGMENT(data)
            _, y_predict = torch.max(y_predict.detach(), dim=1)
            
            # Original Image
            axarr[row, 0].title.set_text("Original Mask")
            axarr[row, 0].imshow(label.cpu().numpy()[0], cmap="gray")
            axarr[row, 0].set_xticks([])
            axarr[row, 0].set_yticks([])
            # Predicted Image
            axarr[row, 1].title.set_text("Predicted Mask")
            axarr[row, 1].imshow(y_predict.cpu().numpy()[0], cmap="gray")
            axarr[row, 1].set_xticks([])
            axarr[row, 1].set_yticks([])
            
            row += 1
            if row >= n_rows:
                break
        
        plt.savefig("outputs/task21_mask_test.pdf", format="pdf", bbox_inches="tight")
