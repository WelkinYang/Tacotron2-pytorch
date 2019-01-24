import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as utils
from hparams import hparams as hp
import numpy as np
from ZoneoutRNN import ZoneoutRNN
import math

class Tacotron(nn.Module):
    def __init__(self, encoder, decoder, postnet, PostCBHG=None, max_length=1000):
        super(Tacotron, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.postnet = postnet
        self.post_cbhg = PostCBHG
        self.max_length = max_length

    @property
    def _stop_at_any(self):
        return hp.stop_at_any

    @property
    def _use_linear_spec(self):
        return hp.use_linear_spec

    @property
    def _use_stop_token(self):
        return hp.use_stop_token

    @property
    def _use_gta_mode(self):
        return hp.use_gta_mode

    def forward(self, input_group, mel_group = None, linear_target=None, stop_token_target=None):
        #input_seqs [batch_size, seq_lens]
        input_seqs, max_input_len = input_group
        mel_target, max_target_len = mel_group
        batch_size = input_seqs.size(0)

        #mel_target [batch_size,  max_target_length / hp.outputs_per_step, decoder_output_size]
        if mel_target is None or max_target_len is None:
            assert hp.use_gta_mode == False, 'if use_gta_mode == True, please provide with target'
            max_target_len = math.ceil(self.max_length / hp.outputs_per_step)
        else:
            max_target_len = math.ceil(max_target_len / hp.outputs_per_step)

        if self._use_gta_mode:
            assert self.training == True, 'When model is evaluating, you can\'t use gta_mode'
        if self._use_linear_spec and self.training:
            assert linear_target is not None, 'When model is training and use_linear_spec is True, ' \
                                              'please apply linear target to calculate loss'
        if self._use_stop_token and self.training:
            assert stop_token_target is not None, 'When model is training and use_stop_token is True, ' \
                                              'please apply stop token target to calculate loss'

        self.encoder.initialize(batch_size, max_input_len)
        encoder_outputs = self.encoder(input_seqs)
        self.decoder.attn.initialize(batch_size, max_input_len, encoder_outputs)
        decoder_inputs = torch.zeros(batch_size, 1, self.decoder.prenet_input_size)
        #initial decoder hidden state
        decoder_hidden = torch.zeros(self.decoder.decoder_lstm_layers, batch_size, self.decoder.decoder_lstm_units)
        decoder_cell_state = torch.zeros(self.decoder.decoder_lstm_layers, batch_size, self.decoder.decoder_lstm_units)
        decoder_outputs = torch.zeros(batch_size, max_target_len, self.decoder.decoder_output_size)
        self.postnet.initialize(self.decoder.decoder_output_size, max_target_len)
        stop_token_prediction = torch.zeros(batch_size, max_target_len, hp.outputs_per_step)

        for t in range(max_target_len):
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
            else:
                decoder_inputs = decoder_outputs[:, t, :]
                finished = (torch.round(stop_token_output) != 0)
                if self._stop_at_any:
                    finished = (torch.sum(finished) > 0)
                else:
                    finished = (torch.sum(finished) == finished.size(1))
                if finished:
                    break

        postnet_outputs = self.postnet(decoder_outputs)
        mel_outputs = decoder_outputs + postnet_outputs

        if self._use_linear_spec:
            self.post_cbhg.initialize(self.decoder.decoder_output_size, batch_size, max_target_len)
            expand_outputs = self.post_cbhg(mel_outputs)
            linear_outputs = F.linear(expand_outputs, weight=torch.nn.init.normal_(torch.empty(hp.num_freq, expand_outputs.shape[2])))

        #calculate losses
        if self.training:

