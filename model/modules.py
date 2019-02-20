import torch
import torch.nn as nn
import torch.nn.functional as F

from text.text import get_vocab_size
from hparams import hparams as hp
from zoneout_rnn import ZoneoutRNN
from model.attention import LocationSensitiveSoftAttention
from utils import Conv1d, highwaynet, compute_same_padding

class EncoderConvlutions(nn.Module):
    def __init__(self, max_length, conv_in_channels, activation=nn.ReLU()):
        super(EncoderConvlutions, self).__init__()
        self.activation = activation
        self.conv1 = nn.Conv1d(conv_in_channels, self.conv_out_channels, kernel_size=self.kernel_size, stride=1, dilation=2,
                               padding=compute_same_padding(self.kernel_size, max_length))
        self.conv2 = nn.Conv1d(self.conv_out_channels, self.conv_out_channels, kernel_size=self.kernel_size, stride=1, dilation=2,
                               padding=compute_same_padding(self.kernel_size, max_length))
        self.batch_norm = nn.BatchNorm1d(self.conv_out_channels)
    @property
    def kernel_size(self):
        return hp.encoder_conv_width

    @property
    def conv_layers(self):
        return hp.encoder_conv_layers

    @property
    def conv_out_channels(self):
        return hp.encoder_conv_channels

    def forward(self, inputs):
        x = inputs
        for i in range(self.conv_layers):
            if i == 0:
                x = Conv1d(x, self.conv1, self.batch_norm, self.training, self.activation)
            else:
                x = Conv1d(x, self.conv2, self.batch_norm, self.training, self.activation)
        outputs = x
        return outputs

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

    def initialize(self, batch_size, max_time):
        self.max_time = max_time
        self.batch_size = batch_size
        self.embedding = nn.Embedding(get_vocab_size(), self.embedding_dim)
        self.encoder_conv = EncoderConvlutions(max_time, self.embedding_dim)
        self.forward_lstm_cell = nn.LSTMCell(self.encoder_conv.conv_out_channels, self.encoder_lstm_units)
        self.backward_lstm_cell = nn.LSTMCell(self.encoder_conv.conv_out_channels, self.encoder_lstm_units)
        self.zoneout_lstm = ZoneoutRNN(self.forward_lstm_cell, self.backward_lstm_cell, (hp.zoneout_prob_cells, hp.zoneout_prob_states))

    @property
    def embedding_dim(self):
        return hp.embedding_dim

    @property
    def encoder_lstm_units(self):
        return hp.encoder_lstm_units

    def forward(self, input_seqs):
        embedded = self.embedding(input_seqs)
        conv_outputs = self.encoder_conv(embedded)
        outputs = torch.zeros([self.batch_size, self.max_time, self.zoneout_lstm.hidden_size * 2])
        forward_h = torch.zeros(self.batch_size, self.zoneout_lstm.hidden_size)
        forward_c = torch.zeros(self.batch_size, self.zoneout_lstm.hidden_size)
        forward_state = (forward_h, forward_c)
        backward_h = torch.zeros(self.batch_size, self.zoneout_lstm.hidden_size)
        backward_c = torch.zeros(self.batch_size, self.zoneout_lstm.hidden_size)
        backward_state = (backward_h, backward_c)
        for i in range(self.max_time):
            forward_input = conv_outputs[:, i, :]
            backward_input = conv_outputs[:, self.max_time - (i + 1), :]
            forward_output, backward_output, \
            forward_new_state, backward_new_state = self.zoneout_lstm(forward_input,
                                                                backward_input,
                                                                forward_state,
                                                                backward_state)
            forward_state = forward_new_state
            backward_state = backward_new_state
            outputs[:, i, :self.zoneout_lstm.hidden_size] = forward_output
            outputs[:, self.max_time - (i + 1),self.zoneout_lstm.hidden_size:] = backward_output

        #output of shape (seq_len, batch, num_directions * hidden_size)
        return outputs

class PreNet(nn.Module):
    def __init__(self, input_size, num_units=256, activation=nn.ReLU()):
        super(PreNet, self).__init__()
        self.num_units = num_units
        self.activation = activation
        self.layer_sizes = layer_sizes=[(input_size, num_units), (num_units, num_units)]
        self.linears = [nn.Linear(sizes[0], sizes[1]) for sizes in self.layer_sizes]
    def forward(self, inputs):
        x = inputs
        for linear in self.linears:
            output = self.activation(linear(x))
            return F.dropout(output, p=hp.dropout_rate, training=self.training)



class Decoder(nn.Module):
    def __init__(self, linear_prejection_activation=None, stop_prejection_activation=nn.Sigmoid()):
        super(Decoder, self).__init__()
        self.prenet = PreNet(self.prenet_input_size)

        self.lstm = nn.LSTM(self.lstm_input_size, self.decoder_lstm_units, self.decoder_lstm_layers, batch_first=True)
        self.attn = LocationSensitiveSoftAttention(self.decoder_lstm_units)
        self.linear_projection = nn.Linear(self.attn.num_units + self.decoder_lstm_units, self.decoder_output_size)
        self.linear_projection_activation = linear_prejection_activation
        self.stop_projection_activation = stop_prejection_activation
        self.stop_projection = nn.Linear(hp.attention_depth + self.decoder_lstm_units, hp.outputs_per_step)

    @property
    def decoder_lstm_units(self):
        return hp.decoder_lstm_units

    @property
    def decoder_lstm_layers(self):
        return hp.decoder_lstm_layers

    @property
    def lstm_input_size(self):
        return self.prenet.num_units + hp.attention_depth

    @property
    def prenet_input_size(self):
        return hp.num_mels * hp.outputs_per_step

    @property
    def decoder_output_size(self):
        return hp.num_mels * hp.outputs_per_step

    def forward(self, input_seqs, last_hidden, last_cell):
        prenet_output = self.prenet(input_seqs)
        last_context_vector = self.attn.context_vector
        last_alignment = self.attn.alignment
        decoder_lstm_inputs = torch.cat((prenet_output, last_context_vector), 2)
        decoder_lstm_outputs, (hn, cn)= self.lstm(decoder_lstm_inputs, (last_hidden, last_cell))
        cur_context_vector = self.attn(decoder_lstm_outputs, last_alignment)
        output = torch.cat((cur_context_vector, decoder_lstm_outputs), 2)
        decoder_output = self.linear_projection(output)
        stop_token_output = self.stop_projection(output)
        if self.linear_projection_activation is not None:
            decoder_output = self.linear_projection_activation(decoder_output)
        if self.stop_projection_activation is not None:
            stop_token_output = self.stop_projection_activation(stop_token_output)
        return decoder_output, stop_token_output, hn, cn, self.attn.alignment

class PostNet(nn.Module):
    def __init__(self):
        super(PostNet, self).__init__()
    def initialize(self, in_channels, max_length, activation=nn.ReLU()):
        self.conv1 = nn.Conv1d(in_channels, self.out_channels, kernel_size=self.kernel_size, stride=1, dilation=2,
                               padding=compute_same_padding(self.kernel_size, max_length))
        self.conv2 = nn.Conv1d(self.out_channels, self.out_channels, kernel_size=self.kernel_size, stride=1, dilation=2,
                               padding=compute_same_padding(self.kernel_size, max_length))
        self.linear = nn.Linear(hp.postnet_conv_channels, in_channels)
        self.batch_norm = nn.BatchNorm1d(self.out_channels)
        self.activation = activation
    @property
    def out_channels(self):
        return hp.postnet_conv_channels

    @property
    def kernel_size(self):
        return hp.postnet_conv_width

    @property
    def layers(self):
        return hp.postnet_conv_layers

    def forward(self, inputs):
        x = inputs
        for i in range(self.layers):
            if i == 0:
                x = Conv1d(x, self.conv1, self.batch_norm, self.training, self.activation)
            else:
                if i < self.layers - 1:
                    x = Conv1d(x, self.conv2, self.batch_norm, self.training, self.activation)
                else:
                    x = Conv1d(x, self.conv2, self.batch_norm, self.training)
        x = self.linear(x)
        return x

class PostCBHG(nn.Module):
    def __init__(self):
        super(PostCBHG, self).__init__()
    def initialize(self, input_channels, input_length, K=8, units=128, activation=[nn.ReLU(), nn.Sigmoid()]):
        self.convs_bank = [nn.Conv1d(in_channels=input_channels, out_channels=units, kernel_size=k, stride=1, dilation=2,
                                padding=compute_same_padding(kernel_size=k, input_length=input_length))
                           for k in range(1, K + 1)]
        self.max_pool = nn.MaxPool1d(kernel_size=2, stride=1, dilation=2,
                                     padding=compute_same_padding(kernel_size=2, input_length=input_length))

        self.conv1 = nn.Conv1d(in_channels=units*K, out_channels=256, kernel_size=3, stride=1, dilation=2,
                               padding=compute_same_padding(kernel_size=3, input_length=input_length))
        self.conv2 = nn.Conv1d(in_channels=256, out_channels=input_channels, kernel_size=3, stride=1, dilation=2,
                               padding=compute_same_padding(kernel_size=3, input_length=input_length))

        self.gru = nn.GRU(units, units, batch_first=True, bidirectional=True)
        self.activation = activation
    def forward(self, inputs):
        # Convolution bank: concatenate on the last axis to stack channels from all convolutions
        conv_outputs = torch.cat([
            Conv1d(inputs, conv, nn.BatchNorm1d(conv.out_channels), self.training, activation=self.activation[0])
            for conv in self.convs_bank], dim=-1)
        # Maxpooling:
        maxpool_output = self.max_pool(conv_outputs)
        # Two projection layers:
        proj1_output = Conv1d(maxpool_output, self.conv1, nn.BatchNorm1d(self.conv1.out_channels), self.training, activation=self.activation[0])
        proj2_output = Conv1d(proj1_output, self.conv2, nn.BatchNorm1d(self.conv2.out_channels), self.training)
        # Residual connection:
        highway_input = proj2_output + inputs

        # Handle dimensionality mismatch:
        if highway_input.shape[2] != 128:
            highway_input = F.linear(highway_input, weight=torch.nn.init.normal_(torch.empty(128, highway_input.shape[2])))

        # 4-layer HighwayNet:
        for i in range(4):
            highway_input = highwaynet(highway_input, self.activation)
        rnn_input = highway_input

        # Bidirectional RNN
        outputs, states = self.gru(rnn_input)
        return outputs
