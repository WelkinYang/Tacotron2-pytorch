import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
from data_utils import get_vocab_size
from hparams import hparams as hp
import numpy as np
from ZoneoutRNN import ZoneoutRNN

class Tacotron(nn.Module):
    def __init__(self, ):
        super(Tacotron, self).__init__()

    def initialize(self, encoder, decoder, postnet, max_length=1000):
        self.encoder = encoder
        self.decoder = decoder
        self.postnet = postnet
        self.max_length = max_length

    def forward(self, input_group, mel_target = None, linear_target=None, stop_token_target=None, target_lens=None):
        #input_seqs [batch_size, seq_lens]
        input_seqs, input_lens = input_group
        batch_size = input_seqs.size(0)
        max_input_len = input_seqs.size(1)
        #mel_target [batch_size,  max_target_length / hp.outputs_per_step, decoder_output_size]
        if mel_target is None or target_lens is None:
            assert hp.use_gta_mode == False, 'if use_gta_mode == True, please provide with target'
            max_target_len = self.max_length / hp.outputs_per_step
        else:
            max_target_len = max(target_lens) / hp.outputs_per_step
        self.encoder.initialize(batch_size, max_input_len)
        encoder_outputs = self.encoder(input_seqs)
        self.decoder.attn.initialize(batch_size, max_target_len, encoder_outputs)
        decoder_inputs = torch.zeros(batch_size, 1, self.decoder.lstm_input_size)
        #initial decoder hidden state
        decoder_hidden = torch.zeros(batch_size, self.decoder.decoder_lstm_layers, self.decoder.decoder_lstm_units)
        decoder_cell = torch.zeros(batch_size, self.decoder.decoder_lstm_layers, self.decoder.decoder_lstm_units)
        decoder_outputs = torch.zeros(batch_size, max_target_len, self.decoder.decoder_output_size)
        self.postnet.initialize(self.decoder.decoder_output_size, max_target_len)
        stop_token_prediction = torch.zeros(batch_size, max_target_len, hp.outputs_per_step)

        for t in range(max_target_len / hp.outputs_per_step):
            decoder_output, stop_token_output, decoder_hidden, decoder_cell_state = \
                self.decoder(decoder_inputs, decoder_hidden, decoder_cell_state)
            decoder_outputs[:, t, :] = torch.squeeze(decoder_output, 1)
            stop_token_prediction[:, t, :] = torch.squeeze(stop_token_output, 1)
            if hp.use_gta_mode:
                if hp.teacher_forcing_schema == "full":
                    decoder_inputs = mel_target[:, t:t+1, :]
                elif hp.teacher_forcing_schema == "semi":
                    decoder_inputs = (
                        decoder_output + mel_target[:, t:t+1, :]
                    ) / 2
                elif hp.teacher_forcing_schema == "random":
                    if np.random.random() <= self.teacher_forcing_ratio:
                        decoder_inputs = mel_target[:, t:t+1, :]
                    else:
                        decoder_inputs = decoder_output
        postnet_outputs = self.postnet(decoder_outputs)
        mel_outputs = decoder_outputs + postnet_outputs

        if hp.use_stop_token:
            stop_token_prediction = torch.reshape(stop_token_prediction, [batch_size, -1])

def _compute_same_padding(kernel_size, input_length):
    #when stride == 1, dilation == 1, groups == 1
    #Lout = [(Lin + 2 * padding - (kernel_size - 1) - 1) + 1] = [Lin + 2 * padding - kernel_size + 1]
    #padding = (Lout + (kernel_size - 1) - Lin) / 2 = (kernel_size - 1) / 2
    return (kernel_size - 1) / 2


class EncoderConvlutions(nn.Module):
    def __init__(self, max_length, conv_layers, conv_in_channels, conv_out_channels, kernel_size, activation=nn.ReLU):
        super(EncoderConvlutions, self).__init__()
        self.activation = activation
        self.conv1 = nn.Conv1d(conv_in_channels, conv_out_channels, kernel_size=self.kernel_size, padding=_compute_same_padding(kernel_size, max_length))
        self.conv2 = nn.Conv1d(conv_out_channels, conv_out_channels, kernel_size=self.kernel_size, padding=_compute_same_padding(kernel_size, max_length))
        self.batch_norm = nn.BatchNorm1d(conv_out_channels)

    def Conv1d(self, inputs, conv):
        conv1d_output = conv(inputs)
        batch_norm_output = self.bath_norm(conv1d_output)
        if self.activation is not None:
            batch_norm_output = self.activation(batch_norm_output)
        return F.dropout(batch_norm_output, p=hp.dropout_rate, training=self.training)
    def forward(self, inputs):
        # the Conv1d of pytroch chanages the channels at the 1 dim
        # [batch_size, max_time, feature_dims] -> [batch_size, feature_dims, max_time]
        x = torch.transpose(inputs, 1, 2)
        for i in range(self.conv_layers):
            if i == 0:
                x = self.Conv1d(x, self.conv1)
            else:
                x = self.Conv1d(x, self.conv2)
        outputs = torch.transpose(x, 1, 2)
        return outputs

class Encoder(nn.Module):
    def __init__(self, max_length):
        super(Encoder, self).__init__()

    def initialize(self, max_time, batch_size):
        self.max_time = max_time
        self.batch_size = batch_size
        self.embedding = nn.Embedding(get_vocab_size(), self.embedding_dim)
        self.encoder_conv = EncoderConvlutions(max_time, self.encoder_conv_layers, self.embedding_dim, self.encoder_conv_channels)
        self.forward_lstm_cell = nn.LSTMCell(self.encoder_conv_channels, self.encoder_lstm_units)
        self.backward_lstm_cell = nn.LSTMCell(self.encoder_conv_channels, self.encoder_lstm_units)
        self.zoneout_lstm = ZoneoutRNN(self.forward_lstm_cell, self.backward_lstm_cell, (hp.zoneout_prob_cells, hp.zoneout_prob_states))
    @property
    def encoder_conv_layers(self):
        return hp.encoder_conv_layers

    @property
    def encoder_conv_channels(self):
        return hp.encoder_conv_channels

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
    def __init__(self, input_size, activation=nn.ReLU):
        super(PreNet, self).__init__()
        self.activation = activation
        self.layer_sizes = layer_sizes=[(input_size, 256), (256, 256)]
        self.linears = [nn.Linear(sizes[0], sizes[1]) for sizes in self.layer_sizes]
    def forward(self, inputs):
        x = inputs
        for linear in self.linears:
            output = self.activation(linear(x))
            return F.dropout(output, p=hp.dropout_rate, training=self.training)

class LocationSensitiveSoftAttention(nn.Module):
    def __init__(self, hidden_size, num_units, cumulate_weights=True):
        super(LocationSensitiveSoftAttention, self).__init__()
        self.num_units = num_units
        self._cumulate = cumulate_weights
        self.query_layer = torch.nn.Linear(hidden_size, num_units)
        self.memoery_layer = torch.nn.Linear(hp.encoder_lstm_units, num_units)
        self.location_layer = torch.nn.Linear(hp.attention_filters, num_units)
        self.location_convolution = nn.Conv1d(hidden_size, hp.attention_filters, kernel_size=hp.kernel_size, padding="same")
        self.compute_energy = nn.Linear(num_units, num_units)
        self.v_a = torch.nn.init.xavier_normal(torch.empty(1, num_units))
    def initialize(self, batch_size, max_time, memoery):
        self.alignment = torch.zeros(batch_size, max_time)
        self.context_vector = torch.zeros(batch_size, 1, self.num_units)
        self.memoery = self.memoery_layer(memoery)

    def  _smoothing_normalization(self, e):
        return torch.sigmoid(e) / torch.sum(torch.sigmoid(e), -1, keepdim=True)
    def forward(self, query, state):
        #query [batch_size, lstm_layers, hidden_size] -> [batch_size, 1, attention_depth(num_units)]
        processed_query = self.query_layer(query)[:, 1:2, :]
        #state [batch_size, max_time] ->  [batch_size, max_time, attention_depth(num_units)]
        #[batch_size, max_time, 1]
        expanded_alignments = torch.unsqueeze(state, 2)
        #[batch_size, max_time, hp.attention_filters]
        f = self.location_convolution(expanded_alignments)
        # [batch_size, max_time, attention_depth(num_units)]
        processed_location_features = self.location_layer(f)

        #energy [batch_size, max_time]
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

class Decoder(nn.Module):
    def __init__(self, linear_prejection_activation=None, stop_prejection_activation=nn.Sigmoid):
        super(Decoder, self).__init__()
        self.prenet = PreNet(self.prenet_input_size)
        self.lstm = nn.LSTM(self.lstm_input_size, self.decoder_lstm_units, self.decoder_lstm_layers, batch_first=True)
        self.attn = LocationSensitiveSoftAttention(self.decoder_lstm_units, hp.attention_depth)
        self.linear_projection = nn.Linear(hp.attention_depth + self.decoder_lstm_layers, self.decoder_output_size)
        self.linear_projection_activation = linear_prejection_activation
        self.stop_projection_activation = stop_prejection_activation
        self.stop_projection = nn.Linear(hp.attention_depth + self.decoder_lstm_layers, hp.outputs_per_step)

    @property
    def decoder_lstm_units(self):
        return hp.decoder_lstm_units

    @property
    def decoder_lstm_layers(self):
        return hp.decoder_lstm_layers

    @property
    def lstm_input_size(self):
        return hp.num_mels * hp.outputs_per_step + hp.attention_depth

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
        output = torch.cat(cur_context_vector, decoder_lstm_outputs, 2)
        decoder_output = self.linear_projection(output)
        stop_token_output = self.stop_projection(output)
        if self.linear_projection_activation is not None:
            decoder_output = self.linear_projection_activation(decoder_output)
        if self.stop_projection_activation is not None:
            stop_token_output = self.linear_projection_activation(stop_token_output)
        return decoder_output, stop_token_output, hn, cn

class PostNet(nn.Module):
    def __init__(self):
        super(PostNet, self).__init__()
    def initialize(self, in_channels, max_length, out_channels=hp.postnet_conv_channels, kernel_size=hp.postnet_conv_width, layers=hp.postnet_conv_layers):
        self.layers = layers
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                               padding=_compute_same_padding(kernel_size, max_length))
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                               padding=_compute_same_padding(kernel_size, max_length))
        self.linear = nn.Linear(hp.postnet_conv_channels, in_channels)
    def Conv1d(self, inputs, cur_layer, conv, activation=None):
        if cur_layer < self.layers - 1:
            activation = torch.tanh
        conv1d_output = conv(inputs)
        batch_norm_output = self.bath_norm(conv1d_output)
        if self.activation is not None:
            batch_norm_output = self.activation(batch_norm_output)
        return F.dropout(batch_norm_output, p=hp.dropout_rate, training=self.training)
    def forward(self, inputs):
        x = inputs
        for i in range(self.layers):
            if i == 0:
                x = self.Conv1d(x, self.conv1)
            else:
                x = self.Conv1d(x, self.conv2)
        x = self.linear(x)
        return x