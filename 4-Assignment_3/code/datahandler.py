"""A module to manage text corpus / dataset."""
import os
from io import open
import torch

class Dictionary(object):
    """A class to keep track of all words / embeddings."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        """A function to add new word to the dictionary."""
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class BatchHandler():
    """A class to generates and manages batches of a dataset."""
    def __init__(self, data_source: torch.Tensor, batch_size: int, sequence_length: int) -> None:
        self.batched_data = self.create_batches(input_data=data_source, batch_size=batch_size)
        self.seq_len = sequence_length
        self._counter = 0

    def __iter__(self):
        self._counter = 0
        return self

    def __next__(self):
        #if self._counter < (self.batched_data.size(0)-1) + self.seq_len:
        next_batch = self.get_batch(self._counter)
        self._counter += self.seq_len
        if len(next_batch[0]) == 0:
            raise StopIteration
        return next_batch
        #else:
        #    raise StopIteration

    # formulate batches based on given batch size
    def create_batches(self, input_data: torch.Tensor, batch_size: int) -> torch.Tensor:
        """A function to create batches of the input data."""
        # find number of batches to be generated
        n_batches = input_data.size(0) // batch_size
        # clean data by dropping last few words
        # that won't fit the batch_size (trimming)
        input_data = input_data.narrow(0, 0, n_batches * batch_size)
        # finaly create batches of the data
        input_data = input_data.view(batch_size, -1).t().contiguous()
        # return the finalized data
        return input_data

    # create a batching routine to get next data batch
    def get_batch(self, start_index):
        """A function to fetch a single batch from the batched data."""
        current_length = min(self.seq_len, len(self.batched_data) - 1 - start_index)
        batch_data = self.batched_data[start_index:start_index+current_length]
        batch_target = self.batched_data[start_index+1:start_index+current_length+1].view(-1)
        return batch_data, batch_target

class Corpus(object):
    """A class to create corpus of text dataset."""
    def __init__(self, root_path: str,
                 sequence_length: int,
                 tr_batchsize: int,
                 vl_batchsize: int,
                 ts_batchsize: int,
                 train_file: str ='ptb.train.txt',
                 test_file: str ='ptb.test.txt',
                 valid_file: str ='ptb.valid.txt',
                 ) -> None:
        # create a dictionary to contain
        # the word to index mappings
        self.dictionary = Dictionary()
        # tokenize each of the input files
        self.train: torch.Tensor = self.tokenize(os.path.join(root_path, train_file))
        self.valid: torch.Tensor = self.tokenize(os.path.join(root_path, valid_file))
        self.test: torch.Tensor = self.tokenize(os.path.join(root_path, test_file))
       # Create batches of the dataset.
        self.train_batches = BatchHandler(self.train, tr_batchsize, sequence_length=sequence_length)
        self.valid_batches = BatchHandler(self.valid, vl_batchsize, sequence_length=sequence_length)
        self.test_batches = BatchHandler(self.test, ts_batchsize, sequence_length=sequence_length)

    def tokenize(self, path) -> torch.Tensor:
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as read_file:
            word_list = []
            for line in read_file:
                words = line.split() + ['<eos>']
                word_list.append(words)
            word2ids_list = [
                self.dictionary.add_word(item) for sublist in word_list for item in sublist
            ]
            return torch.Tensor(word2ids_list).type(torch.int64)

if __name__ == "__main__":
    corpus = Corpus("./dataset/data/", 5, 5, 2, 2)
    print(len(corpus.dictionary))

    for data, target in corpus.train_batches:
        print(data, target)
