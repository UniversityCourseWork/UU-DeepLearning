"""Main module of the code that brings everything together."""

import torch
from torch import nn
import torch.nn.functional as F
import math

from datahandler import Corpus
from modelhandler import SimpleRNN


def repackage_hidden(hidden):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(hidden, torch.Tensor):
        return hidden.detach()
    else:
        return tuple(repackage_hidden(v) for v in hidden)


def predict(model, corpus, n_words, device, temp=1.0):
    """A function to handle the predictions from the RNN."""
    # put the model in evaluation mode
    model.eval()

    input_words = torch.randint(len(corpus.dictionary), (1, 1), dtype=torch.long).to(device)
    sentence = []
    
    hidden = model.init_hidden(1)
    with torch.no_grad():
        for i in range(n_words):
            output, hidden = model(input_words, hidden)
            # word_idx = torch.argmax(output, dim=1).cpu().item()
            word_weights = output.squeeze().div(temp).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            input_words.fill_(word_idx)
            sentence.append(corpus.dictionary.idx2word[word_idx])
        print(sentence)


def evaluate(model, criterion, data_source, batch_size, device):
    """A function to handle the evaluation routine of the RNN."""
    # put the model in evaluation mode
    model.eval()

    total_loss = 0.0
    hidden = model.init_hidden(batch_size=batch_size)

    # perform evaluation
    with torch.no_grad():
        for batch in data_source:
            data, targets = batch
            data, targets = data.to(device), targets.to(device)
            output, hidden = model(data, hidden)
            hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()

    # return average loss
    return total_loss / (len(data_source)-1)


def train(model, device, criterion, optimizer, data_source, batch_size, lr, clip=0.25):
    """A function to handle the training routine of the RNN."""
    # prepare model for training
    model.train()
    total_loss = 0.0
    grand_total_loss = 0.0
    hidden = model.init_hidden(batch_size=batch_size)

    for indx, batch in enumerate(data_source):
        data, targets = batch
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        # model.zero_grad()
        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        #torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        #for p in model.parameters():
        #    p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()
        grand_total_loss += loss.item()

        # printout a log message
        if indx % 200 == 0 and indx > 0:
            current_loss = total_loss / 200
            print(f"| {indx:5d}  / {data_source.n_batches//data_source.seq_len:5d} batches | lr = {lr:02.2f} | train loss = {current_loss:5.4f} | train perplexity = {math.exp(current_loss):20.2f}")
            total_loss = 0.0

    return grand_total_loss


def train_model(criterion,
                data_path: str,
                tr_bs: float,
                vl_bs: float,
                ts_bs: float,
                epochs: int,
                seq_len: int,
                embedd_dim: int,
                hidden_dim: int,
                num_layers: int,
                learn_rate: float,
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    best_val_loss = None

    try:
        # train the model
        for epoch in range(epochs):
            epoch_loss = train(model=model,
                               device=device,
                               criterion=criterion,
                               optimizer=optimizer,
                               data_source=corpus.train_batches,
                               batch_size=tr_bs,
                               lr=learn_rate)
            val_loss = evaluate(model=model,
                                data_source=corpus.valid_batches,
                                batch_size=vl_bs,
                                criterion=criterion,
                                device=device)
            print('-' * 105)
            print(f"|   End of Epoch {epoch+1:4d} of {epochs:4d}   ||   Valid Loss = {val_loss:5.2f}   ||   Valid Perplexity = {math.exp(val_loss):20.2f}")
            print('-' * 105)

            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open("./checkpoints/best_model.pt", 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # Reduce learning rate
                learn_rate /= 1.5
    except KeyboardInterrupt:
        pass
    
    # evaluate the model on test dataset
    test_loss = evaluate(model=model,
                         data_source=corpus.test_batches,
                         batch_size=ts_bs,
                         criterion=criterion,
                         device=device)
    # print out final statistics on test data
    print('=' * 105)
    print(f"|     End of Training     ||     Test Loss = {test_loss:5.2f}     ||     Test Perplexity = {math.exp(test_loss):20.2f}")
    print('=' * 105)

    # test prediction function
    predict(model, corpus=corpus, n_words=100, device=device)


if __name__ == "__main__":
    # Global Variables that contain all the
    # hyperparameters in central location
    DATA_PATH = "./dataset/data/"
    BATCH_SIZE = 20
    SEQ_LEN = 35
    EMBEDD_DIM = 200
    HIDDEN_DIM = 200
    NUM_LAYERS = 4
    EPOCHS = 100
    # CRITERION = nn.NLLLoss()
    CRITERION = nn.CrossEntropyLoss()
    OPTIM_NAME = "SGD"
    LEARN_RATE = 0.1
    SEED = 1111

    torch.manual_seed(SEED)

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
