import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
from data_utils import get_vocab_size
from hparams import hparams as hp
import numpy as np

class Tacotron(nn.Module):
    def __init__(self, ):
        super(Tacotron, self).__init__()

    def initialize(self, encoder, decoder, postnet, max_length=1000):
        self.encoder = encoder
        self.decoder = decoder
        self.postnet = PostNet
        self.max_length = max_length

    def forward(self, input_group, mel_target = None, linear_target=None, stop_token_target=None, target_lens=None):
        #input_seqs [batch_size, seq_lens]
        input_seqs, input_lens = input_group
        batch_size = input_seqs.size(0)
        #mel_target [batch_size,  max_target_length / hp.outputs_per_step, decoder_output_size]
        if mel_target is None or target_lens is None:
            assert hp.use_gta_mode == False, 'if use_gta_mode == True, please provide with target'
            max_target_length = self.max_length / hp.outputs_per_step
        else:
            max_target_length = max(target_lens) / hp.outputs_per_step
        self.encoder.initialize(input_lens)
        encoder_outputs = self.encoder(input_seqs, input_lens)
        self.decoder.attn.initialize(batch_size, max_target_length, encoder_outputs)
        decoder_inputs = torch.zeros(batch_size, 1, self.decoder.lstm_input_size)
        #initial decoder hidden state
        decoder_hidden = torch.zeros(batch_size, hp.decoder_lstm_layers, hp.decoder_lstm_units)
        decoder_cell = torch.zeros(batch_size, hp.decoder_lstm_layers, hp.decoder_lstm_units)
        decoder_outputs = torch.zeros(batch_size, max_target_length, self.decoder.decoder_output_size)
        self.postnet.initialize(self.decoder.decoder_output_size)
        stop_token_prediction = torch.zeros(batch_size, max_target_length, hp.outputs_per_step)

        for t in range(max_target_length / hp.outputs_per_step):
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

    def initialize(self, max_length):
        self.embedding = nn.Embedding(get_vocab_size(), hp.embedding_dim)
        self.encoder_conv = EncoderConvlutions(max_length, hp.encoder_conv_layers, hp.embedding_dim, hp.encoder_conv_channels)
        self.lstm = nn.LSTM(hp.encoder_conv_channels, hp.encoder_lstm_units, batch_first=True, bidirectional=True)
        self.hidden_size = hp.encoder_lstm_units

    def forward(self, input_seqs, input_lens, h_0=None, c_0=None):
        embedded = self.embedding(input_seqs)
        conv_outputs = self.encoder_conv(embedded)
        packed = utils.pack_padded_sequence(conv_outputs, input_lens, batch_first=True)
        outputs, (hn, cn) = self.lstm(packed, (h_0, c_0))
        outputs, output_lengths = utils.pad_packed_sequence(outputs)
        #output of shape (seq_len, batch, num_directions * hidden_size)
        outputs = outputs[:, :, self.hidden_size] + outputs[:, :, self.hidden_size:]
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
    def __init__(self, prenet_input_size = hp.num_mels * hp.outputs_per_step, lstm_input_size = (hp.num_mels * hp.outputs_per_step + hp.attention_depth),
                 linear_prejection_activation=None, stop_prejection_activation=nn.Sigmoid):
        super(Decoder, self).__init__()
        self.lstm_input_size = lstm_input_size
        self.decoder_output_size = prenet_input_size
        self.prenet = PreNet(prenet_input_size)
        self.lstm = nn.LSTM(lstm_input_size, hp.decoder_lstm_units, hp.decoder_lstm_layers, batch_first=True)
        self.attn = LocationSensitiveSoftAttention(hp.deocder_lstm_units, hp.attention_depth)
        self.linear_projection = nn.Linear(hp.attention_depth + hp.decoder_lstm_layers, self.decoder_output_size)
        self.linear_projection_activation = linear_prejection_activation
        self.stop_projection_activation = stop_prejection_activation
        self.stop_projection = nn.Linear(hp.attention_depth + hp.decoder_lstm_layers, hp.outputs_per_step)
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
    def initialize(self, in_channels, out_channels=hp.postnet_conv_channels, kernel_size=hp.postnet_conv_width, layers=hp.postnet_conv_layers):
        self.layers = layers
        self.conv = nn.Conv1d(in_channels, hp.postnet_conv_channels, kernel_size, padding="same")
        self.linear = nn.Linear(hp.postnet_conv_channels, in_channels)
    def Conv1d(self, inputs, activation=None):
        conv1d_output = self.conv(inputs)
        batch_norm_output = self.bath_norm(conv1d_output)
        if self.activation is not None:
            batch_norm_output = self.activation(batch_norm_output)
        return F.dropout(batch_norm_output, p=hp.dropout_rate, training=self.training)
    def forward(self, inputs):
        x = inputs
        for i in range(self.layers):
            if i < self.layers - 1:
                x = self.Conv1d(x, torch.tanh)
            else:
                x = self.Conv1d(x)
        x = self.linear(x)
        return x