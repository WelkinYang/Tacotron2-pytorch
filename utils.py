import torch
import torch.nn as nn
import torch.functional as F
from hparams import hparams as hp

def make_divisible(dividend, divisor):
    return dividend if dividend % divisor == 0 \
            else (dividend + divisor - dividend % divisor)

def compute_same_padding(kernel_size, input_length, dilation=2):
    #when stride == 1, dilation == 2, groups == 1
    #Lout = [(Lin + 2 * padding - dilation * (kernel_size - 1) - 1) + 1]
    #padding = dilation * (kernel_size - 1) / 2
    return int(dilation * (kernel_size - 1) / 2)

def Conv1d(inputs, conv, batch_norm, is_training, activation=None):
    # the Conv1d of pytroch chanages the channels at the 1 dim
    # [batch_size, max_time, feature_dims] -> [batch_size, feature_dims, max_time]
    inputs = torch.transpose(inputs, 1, 2)
    conv1d_output = conv(inputs)
    batch_norm_output = batch_norm(conv1d_output)
    batch_norm_output = torch.transpose(batch_norm_output, 1 ,2)
    if activation is not None:
        batch_norm_output = activation(batch_norm_output)
    return F.dropout(batch_norm_output, p=hp.dropout_rate, training=is_training)

def highwaynet(inputs, activation, units=128):
    H = F.linear(inputs, weight=torch.nn.init.normal_(torch.empty(units, inputs.size(2))))
    H = activation[0](H)
    T = F.linear(inputs, weight=torch.nn.init.normal_(torch.empty(units, inputs.size(2))), bias=nn.init.constant_(torch.empty(1, 1, units), -0.1))
    T = activation[1](T)
    return H * T + inputs * (1.0 - T)


