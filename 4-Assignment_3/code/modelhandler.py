"""A module to manage RNN model."""

from torch import nn
import torch.nn.functional as F

# create a simple RNN model
class SimpleRNN(nn.Module):
    """Simple Recurrent Neural Network implementation."""
    def __init__(self, num_tokens, embedding_dim, hidden_dim, num_layers, dropout=0.5):
        super(SimpleRNN, self).__init__()
        # save parameters as class variables
        self.num_tokens = num_tokens
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # create an embedding layer
        self.embedding_layers = nn.Embedding(num_tokens, embedding_dim)
        # create hidden RNN layers
        self.recurrent_layers = nn.RNN(embedding_dim, hidden_dim, num_layers, dropout=dropout)
        # create output layer
        #self.output_layers = nn.Linear(in_features=hidden_dim, out_features=num_tokens)
        self.output_layers = nn.Linear(hidden_dim, num_tokens)
        # initialize weights
        self.init_weights()

    def forward(self, input_x, hidden):
        """Forward propagation routine."""
        # apply embedding layer first
        embedded = self.embedding_layers(input_x)
        # apply the hidden RNN layers next
        output, hidden = self.recurrent_layers(embedded, hidden)
        # run the output through
        # the final output layers
        output = self.output_layers(output)
        # apply the softmax function on output
        output = output.view(-1, self.num_tokens)
        output = F.log_softmax(output, dim=1)
        # return the output generated
        return output, hidden

    def init_weights(self):
        """Weight initialization routine."""
        initrange = 0.1
        nn.init.uniform_(self.embedding_layers.weight, -initrange, initrange)
        nn.init.zeros_(self.output_layers.bias)
        nn.init.uniform_(self.output_layers.weight, -initrange, initrange)

    def init_hidden(self, batch_size):
        """Hidden unit initialization routine."""
        weight = next(self.parameters())
        return weight.new_zeros(self.num_layers, batch_size, self.hidden_dim)
        #return (weight.new_zeros(self.num_layers, batch_size, self.hidden_dim),
        #        weight.new_zeros(self.num_layers, batch_size, self.hidden_dim))
