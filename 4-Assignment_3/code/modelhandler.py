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
        # create a dropout layer
        self.dropout_layer = nn.Dropout(dropout)
        # create an embedding layer
        self.embedding_layer = nn.Embedding(num_tokens, embedding_dim)
        # create hidden RNN layers
        self.recurrent_layer = nn.RNN(embedding_dim, hidden_dim, num_layers, dropout=dropout, nonlinearity="tanh")
        # create output layer
        self.output_layer = nn.Linear(hidden_dim, num_tokens)
        # initialize weights
        self.init_weights()

    def forward(self, input_x, hidden):
        """Forward propagation routine."""
        # apply embedding layer first
        # embedded = self.dropout_layer(self.embedding_layer(input_x))
        embedded = self.embedding_layer(input_x)
        # apply the hidden RNN layers next
        output, hidden = self.recurrent_layer(embedded, hidden)
        # output = self.dropout_layer(output)
        # run the output through
        # the final output layers
        output = self.output_layer(output)
        # print(output.shape)
        # apply the softmax function on output
        output = output.view(-1, self.num_tokens)
        # print(output.shape)
        # output = F.softmax(output, dim=1) #F.log_softmax(output, dim=1)
        # print(output.shape)
        # return the output generated
        return output, hidden

    def init_weights(self):
        """Weight initialization routine."""
        initrange = 0.1
        nn.init.uniform_(self.embedding_layer.weight, -initrange, initrange)
        nn.init.zeros_(self.output_layer.bias)
        nn.init.uniform_(self.output_layer.weight, -initrange, initrange)

    def init_hidden(self, batch_size):
        """Hidden unit initialization routine."""
        weight = next(self.parameters())
        return weight.new_zeros(self.num_layers, batch_size, self.hidden_dim)
        #return (weight.new_zeros(self.num_layers, batch_size, self.hidden_dim),
        #        weight.new_zeros(self.num_layers, batch_size, self.hidden_dim))
