import torch
import torch.nn as nn
from hparams import hparams as hp
from modules import compute_same_padding

class LocationSensitiveSoftAttention(nn.Module):
    def __init__(self, hidden_size, cumulate_weights=True):
        super(LocationSensitiveSoftAttention, self).__init__()
        self._cumulate = cumulate_weights
        self.query_layer = torch.nn.Linear(hidden_size, self.num_units)
        self.memoery_layer = torch.nn.Linear(hp.encoder_lstm_units*2, self.num_units)
        self.location_layer = torch.nn.Linear(hp.attention_filters, self.num_units)

        self.compute_energy = nn.Linear(self.num_units, self.num_units)
        self.v_a = torch.nn.init.xavier_normal_(torch.empty(1, self.num_units))

    def initialize(self, batch_size, max_time, memoery):
        self.max_time = max_time
        self.alignment = torch.zeros(batch_size, max_time)
        self.context_vector = torch.zeros(batch_size, 1, self.num_units)
        self.memoery = self.memoery_layer(memoery)
        self.location_convolution = nn.Conv1d(1, hp.attention_filters, kernel_size=hp.attention_kernel, stride=1, dilation=2,
                                              padding=compute_same_padding(self.kernel_size, self.max_time))

    @property
    def kernel_size(self):
        return hp.attention_kernel
    @property
    def num_units(self):
        return hp.attention_depth

    def  _smoothing_normalization(self, e):
        return torch.sigmoid(e) / torch.sum(torch.sigmoid(e), -1, keepdim=True)

    def forward(self, query, state):
        #query [batch_size, lstm_layers, hidden_size] -> [batch_size, 1, attention_depth(num_units)]
        processed_query = self.query_layer(query)[:, 0:1, :]
        #state [batch_size, max_time] ->  [batch_size, max_time, attention_depth(num_units)]
        #[batch_size, max_time, 1]
        expanded_alignments = torch.unsqueeze(state, 2)
        #[batch_size, 1, max_time] transpose because of the strange conv api
        expanded_alignments = torch.transpose(expanded_alignments, 1, 2)
        #[batch_size, hp.attention_filters, max_time]
        f = self.location_convolution(expanded_alignments)
        #[batch_size, max_time, hp.attention_filters]
        f = torch.transpose(f, 1, 2)
        # [batch_size, max_time, attention_depth(num_units)]
        processed_location_features = self.location_layer(f)

        #energy [batch_size, max_time]
        tmp = processed_query + processed_location_features + self.memoery
        energy = torch.sum(self.v_a * torch.tanh(self.compute_energy(processed_query + processed_location_features + self.memoery)), 2)
        alignment = self._smoothing_normalization(energy)

        if self._cumulate:
            self.alignment += alignment
        else:
            self.alignment = alignment
        #expanded_alignments [batch_size, 1, max_time]
        expanded_alignments = torch.unsqueeze(self.alignment, 1)
        #[batch_size, 1, max_time] * [batch_size, max_time, attention_depth] = [batch_size, 1, attention_depth]
        self.context_vector = torch.matmul(expanded_alignments, self.memoery)
        return self.context_vector