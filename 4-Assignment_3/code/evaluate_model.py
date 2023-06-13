"""Main module of the code that brings everything together."""

import os
import math
import datetime
import random

import torch
import torch.nn.functional as F
from torch import nn

from configs import parse_configs
from datahandler import Corpus

def repackage_hidden(hidden):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(hidden, torch.Tensor):
        return hidden.detach()
    else:
        return tuple(repackage_hidden(v) for v in hidden)


def predict(model, corpus, n_words, device, temp=1.0, initial_phrase = None):
    """A function to handle the predictions from the RNN."""
    # put the model in evaluation mode
    model.eval()

    if initial_phrase is not None:
        indxes = [corpus.dictionary.word2idx[word] for word in initial_phrase.split()]
    else:
        indxes = [random.randint(0, len(corpus.dictionary))]
    
    sentence = [corpus.dictionary.idx2word[idx] for idx in indxes]
    
    hidden = model.init_hidden(1)
    with torch.no_grad():
        for i in range(n_words):
            input_words = torch.tensor([indxes], dtype=torch.long).T.to(device)
            output, hidden = model(input_words, hidden)
            #print(output.shape)
            word_weights = output.squeeze().div(temp).exp().cpu()
            #word_idx = torch.argmax(word_weights, dim=1)[3]
            word_idx = torch.multinomial(word_weights, num_samples=1)[0]
            #probs = torch.softmax(output[:, -1] / temp, dim=-1)  
            #word_idx = torch.multinomial(probs, num_samples=1).item()
            indxes.append(word_idx)
            sentence.append(corpus.dictionary.idx2word[word_idx])
        print(" ".join(sentence))


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

def evaluate_model(model_path, model_type, user_options, gen_text = False):
    """A function to handle the training routine of the RNN."""
    model = torch.load(model_path)
        
    corpus = Corpus(root_path=user_options["DATASET_PATH"],
                    sequence_length=user_options["SEQUEQNCE_LENGTH"],
                    tr_batchsize=user_options["BATCH_SIZE"],
                    vl_batchsize=user_options["BATCH_SIZE"],
                    ts_batchsize=user_options["BATCH_SIZE"])

    # create a loss function
    criterion = nn.NLLLoss() if user_options["CRITERION"] == "NLLLOSS" else nn.CrossEntropyLoss()

    # check which device to run the code on
    # use GPU whenever it is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.model_type = model_type
    model.to(device=device)
    model.eval()
    
    # evaluate the model on test dataset
    test_loss = evaluate(model=model,
                         data_source=corpus.test_batches,
                         batch_size=user_options["BATCH_SIZE"],
                         criterion=criterion,
                         device=device)

    # print out final statistics on test data
    print('=' * 125)
    print(f"{model_path}     ||     Test Loss = {test_loss:5.2f}     ||     Test Perplexity = {math.exp(test_loss):0.2f}")

    # test prediction function
    if gen_text:
        predict(model, corpus=corpus, n_words=200, device=device, initial_phrase = "once again the specialists")


if __name__ == "__main__":
    eval_configs = parse_configs("./configs/eval_configs.yml")
    indx = 0
    for subdir, dirs, files in os.walk(eval_configs["CHECKPOINT_PATH"]):
        for file in files:
            model_index = int(file[3:5])
            gen_text = (model_index == 23 or model_index == 33)
            if not gen_text:
                continue 
            model_type = "RNN" if model_index < 20 else "LSTM"
            evaluate_model(model_path=os.path.join(subdir, file), model_type=model_type, user_options=eval_configs, gen_text=gen_text)
            indx += 1
    print('=' * 125)
