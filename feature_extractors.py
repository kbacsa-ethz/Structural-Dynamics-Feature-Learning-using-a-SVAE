import numpy as np
import torch
import torch.nn as nn


# https://discuss.pytorch.org/t/how-does-the-batch-normalization-work-for-sequence-data/30839/11
# Define a utility function for applying a pointwise function to a packed sequence
def simple_elementwise_apply(fn, packed_sequence):
    """Applies a pointwise function fn to each element in packed_sequence"""
    return torch.nn.utils.rnn.PackedSequence(fn(packed_sequence.data), packed_sequence.batch_sizes)


# Define a class for a LSTM-based feature extractor with single output
class LSTMSingleOutput(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layers, dropout, seq_len, batch_size):
        super(LSTMSingleOutput, self).__init__()

        # Mask to make the length of sequences variable between seq_len and 1/3
        seq_lengths = list(np.linspace(seq_len // 6, seq_len, batch_size).astype(np.int))[::-1]
        mask = torch.zeros([batch_size, seq_len, in_channels])
        for i in range(batch_size):
            mask[i, (seq_len - seq_lengths[i]) // 2:(-seq_len + seq_lengths[i]) // 2, :] = 1
        self.mask = mask.float()
        self.seq_lengths = torch.LongTensor(seq_lengths)

        # LSTM layer
        self.lstm = torch.nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout
        )

        # Batch normalization layer
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        # Horizon invariant (applies the mask during training)
        if self.training:
            x *= self.mask[:x.size(0)].to(x.device)
            x = torch.nn.utils.rnn.pack_padded_sequence(x, batch_first=True,
                                                        lengths=self.seq_lengths[0] * torch.ones_like(self.seq_lengths)[
                                                                                      :x.size(0)])
            x, _ = self.lstm(x)
            x = simple_elementwise_apply(self.bn, x)
            x, lengths = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        else:
            x, _ = self.lstm(x)
            x = self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


# Define a class for a vanilla RNN-based feature extractor with single output
class RNNSingleOutput(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layers, dropout, seq_len, batch_size):
        super(RNNSingleOutput, self).__init__()

        # Mask to make the length of sequences variable between seq_len and 1/3
        seq_lengths = list(np.linspace(seq_len // 3, seq_len, batch_size).astype(np.int))[::-1]
        mask = torch.zeros([batch_size, seq_len, in_channels])
        for i in range(batch_size):
            mask[i, (seq_len - seq_lengths[i]) // 2:(-seq_len + seq_lengths[i]) // 2, :] = 1
        self.mask = mask.float()
        self.seq_lengths = torch.LongTensor(seq_lengths)

        # RNN layer
        self.rnn = torch.nn.RNN(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout
        )

        # Batch normalization layer
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        # Horizon invariant (applies the mask during training)
        if self.training:
            x *= self.mask[:x.size(0)].to(x.device)
            x = torch.nn.utils.rnn.pack_padded_sequence(x, batch_first=True,
                                                        lengths=self.seq_lengths[0] * torch.ones_like(self.seq_lengths)[
                                                                                      :x.size(0)])
            x, _ = self.rnn(x)
            x = simple_elementwise_apply(self.bn, x)
            x, lengths = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        else:
            x, _ = self.rnn(x)
            x = self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


# Define a class for a vanilla RNN-based feature extractor with single output and no mask
class RNNSingleOutputNoMask(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layers, dropout):
        super(RNNSingleOutputNoMask, self).__init__()

        # RNN layer
        self.rnn = torch.nn.RNN(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout
        )

        # Batch normalization layer
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x


# Define a class for a GRU-based feature extractor with single output
class GRUSingleOutput(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layers, dropout, seq_len, batch_size):
        super(GRUSingleOutput, self).__init__()

        # Mask to make the length of sequences variable between seq_len and 1/3
        seq_lengths = list(np.linspace(seq_len // 3, seq_len, batch_size).astype(np.int))[::-1]
        mask = torch.zeros([batch_size, seq_len, in_channels])
        for i in range(batch_size):
            mask[i, (seq_len - seq_lengths[i]) // 2:(-seq_len + seq_lengths[i]) // 2, :] = 1
        self.mask = mask.float()
        self.seq_lengths = torch.LongTensor(seq_lengths)

        # GRU layer
        self.gru = torch.nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bias=True,
            batch_first=True,
            dropout=dropout
        )

        # Batch normalization layer
        self.bn = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        # Horizon invariant (applies the mask during training)
        if self.training:
            x *= self.mask[:x.size(0)].to(x.device)
            x = torch.nn.utils.rnn.pack_padded_sequence(x, batch_first=True,
                                                        lengths=self.seq_lengths[0] * torch.ones_like(self.seq_lengths)[
                                                                                      :x.size(0)])
            x, _ = self.gru(x)
            x = simple_elementwise_apply(self.bn, x)
            x, lengths = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        else:
            x, _ = self.gru(x)
            x = self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x
