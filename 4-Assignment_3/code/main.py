"""Main module of the code that brings everything together."""

import torch
from torch import nn

from datahandler import Corpus
from modelhandler import SimpleRNN


def repackage_hidden(hidden):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(hidden, torch.Tensor):
        return hidden.detach()
    else:
        return tuple(repackage_hidden(v) for v in hidden)


def evaluate(model, criterion, data_source, batch_size):
    """A function to handle the evaluation routine of the RNN."""
    # put the model in evaluation mode
    model.eval()

    total_loss = 0.0
    hidden = model.init_hidden(batch_size=batch_size)

    # perform evaluation
    with torch.no_grad():
        for batch in data_source:
            data, targets = batch
            output, hidden = model(data, hidden)
            hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()

    # return average loss
    return total_loss / (len(data_source)-1)


def train(model, device, criterion, optimizer, data_source, batch_size):
    """A function to handle the training routine of the RNN."""
    # prepare model for training
    model.train()
    total_loss = 0.0
    hidden = model.init_hidden(batch_size=batch_size)

    for batch in data_source:
        data, targets = batch
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss


def train_model(criterion,
                epochs,
                data_path,
                seq_len,
                tr_bs,
                vl_bs,
                ts_bs,
                embedd_dim,
                hidden_dim,
                num_layers,
                learn_rate: float = 0.01,
                optim_name: str = "SGD",
                model_name: str = "RNN"):
    """A function to handle the training routine of the RNN."""
    corpus = Corpus(root_path=data_path,
                    sequence_length=seq_len,
                    tr_batchsize=tr_bs,
                    vl_batchsize=vl_bs,
                    ts_batchsize=ts_bs)

    # check which device to run the code on
    # use GPU whenever it is available
    device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if model_name == "RNN":
        model = SimpleRNN(num_tokens=len(corpus.dictionary),
                                embedding_dim=embedd_dim,
                                hidden_dim=hidden_dim,
                                num_layers=num_layers)

    if optim_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)
    elif optim_name == "ADAM":
        optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    model.to(device=device)

    # train the model
    for epoch in range(epochs):
        epoch_loss = train(model=model,
                                        device=device,
                                        criterion=criterion,
                                        optimizer = optimizer,
                                        data_source=corpus.train_batches,
                                        batch_size=tr_bs)

        print(f"EPOCH {epoch+1} of {epochs}: Loss = {epoch_loss}.")


if __name__ == "__main__":
    # Global Variables that contain all the
    # hyperparameters in central location
    DATA_PATH = "./dataset/data/"
    BATCH_SIZE = 20
    SEQ_LEN = 35
    EMBEDD_DIM = 200
    HIDDEN_DIM = 200
    NUM_LAYERS = 2
    EPOCHS = 100
    CRITERION = nn.CrossEntropyLoss()
    OPTIM_NAME = "SGD"
    LEARN_RATE = 20

    train_model(model_name="RNN",
                optim_name=OPTIM_NAME,
                learn_rate=LEARN_RATE,
                criterion=CRITERION,
                epochs=EPOCHS,
                data_path=DATA_PATH,
                seq_len=SEQ_LEN,
                tr_bs=BATCH_SIZE,
                ts_bs=BATCH_SIZE,
                vl_bs=BATCH_SIZE,
                embedd_dim=EMBEDD_DIM,
                hidden_dim=HIDDEN_DIM,
                num_layers=NUM_LAYERS)
