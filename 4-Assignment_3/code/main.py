"""Main module of the code that brings everything together."""

import os
import math
import datetime

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from configs import parse_configs
from datahandler import Corpus
from modelhandler import SimpleRNN
#import numpy as np

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


def train(model, device, criterion, optimizer, data_source, batch_size, lr, hidden, clip_norm=False, clip_limit=0.25):
    """A function to handle the training routine of the RNN."""
    # prepare model for training
    model.train()
    total_loss = 0.0
    grand_total_loss = 0.0
    # hidden = model.init_hidden(batch_size=batch_size)

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
        if clip_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_limit)
            for p in model.parameters():
                p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()
        grand_total_loss += loss.item()

        # printout a log message
        if indx % 200 == 0 and indx > 0:
            current_loss = total_loss / 200
            print(f"| {indx:5d}  / {data_source.n_batches//data_source.seq_len:5d} batches | lr = {lr:02.2f} | train loss = {current_loss:5.4f} | train perplexity = {math.exp(current_loss):20.2f}")
            total_loss = 0.0

    return grand_total_loss / indx


def train_model():
    """A function to handle the training routine of the RNN."""
    global user_options
    torch.manual_seed(user_options["RANDOM_SEED"])

    corpus = Corpus(root_path=user_options["DATASET_PATH"],
                    sequence_length=user_options["SEQUEQNCE_LENGTH"],
                    tr_batchsize=user_options["TRAIN_BATCH_SIZE"],
                    vl_batchsize=user_options["VALID_BATCH_SIZE"],
                    ts_batchsize=user_options["TESTS_BATCH_SIZE"])

    # check which device to run the code on
    # use GPU whenever it is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if user_options["MODEL_TYPE"] == "RNN":
        model = SimpleRNN(num_tokens=len(corpus.dictionary),
                          embedding_dim=user_options["EMBEDDEDING_SIZE"],
                          hidden_dim=user_options["NUM_HIDDEN_UNITS"],
                          num_layers=user_options["NUM_LAYERS"])

    if user_options["OPTIMIZER_NAME"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=user_options["LEARN_RATE"])
    elif user_options["OPTIMIZER_NAME"] == "ADAM":
        optimizer = torch.optim.Adam(model.parameters(), lr=user_options["LEARN_RATE"])

    model.to(device=device)
    best_val_loss = None
    # create a loss function
    criterion = nn.NLLLoss() if user_options["CRITERION"] == "NLLLOSS" else nn.CrossEntropyLoss()

    # create a tensorboard writer to keep track of stats
    tbWriter = SummaryWriter(
        log_dir=user_options["SUMMARY_PATH"] + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    try:
        # initialize hidden state
        hidden = model.init_hidden(batch_size=user_options["TRAIN_BATCH_SIZE"])
        # train the model
        for epoch in range(user_options["EPOCHS"]):
            
            if not user_options["REUSE_HIDDEN"]:
                hidden = model.init_hidden(batch_size=user_options["TRAIN_BATCH_SIZE"])
            
            train_loss = train(model=model,
                               device=device,
                               criterion=criterion,
                               optimizer=optimizer,
                               data_source=corpus.train_batches,
                               batch_size=user_options["TRAIN_BATCH_SIZE"],
                               lr=user_options["LEARN_RATE"],
                               hidden=hidden,
                               clip_norm=user_options["CLIP_GRADIENT"],
                               clip_limit=user_options["CLIP_LIMIT"])
            
            val_loss = evaluate(model=model,
                                data_source=corpus.valid_batches,
                                batch_size=user_options["VALID_BATCH_SIZE"],
                                criterion=criterion,
                                device=device)

            tbWriter.add_scalar("Train_Loss", train_loss, epoch)
            tbWriter.add_scalar("Valid_Loss", val_loss, epoch)
            tbWriter.add_scalar("Train_PPL", math.exp(train_loss), epoch)
            tbWriter.add_scalar("Valid_PPL", math.exp(val_loss), epoch)

            print('-' * 105)
            print(f"|   End of Epoch {epoch+1:4d} of {user_options['EPOCHS']:4d}   ||   Valid Loss = {val_loss:5.2f}   ||   Valid Perplexity = {math.exp(val_loss):20.2f}")
            print('-' * 105)

            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                os.makedirs(os.path.dirname(user_options['CHECKPOINT_PATH']), exist_ok=True)
                with open(user_options['CHECKPOINT_PATH'], 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # Reduce learning rate
                user_options["LEARN_RATE"] /= 1.5
    except KeyboardInterrupt:
        print("|     Keyboard Interrupt, ending model training")
    
    tbWriter.close()
    
    # evaluate the model on test dataset
    test_loss = evaluate(model=model,
                         data_source=corpus.test_batches,
                         batch_size=user_options["TESTS_BATCH_SIZE"],
                         criterion=criterion,
                         device=device)
    
    # print out final statistics on test data
    print('=' * 105)
    print(f"|     End of Training     ||     Test Loss = {test_loss:5.2f}     ||     Test Perplexity = {math.exp(test_loss):20.2f}")
    print('=' * 105)

    # test prediction function
    predict(model, corpus=corpus, n_words=100, device=device)


if __name__ == "__main__":
    global user_options
    user_options = parse_configs("./configs/config.yml")
    train_model()
