import torch
import pandas as pd
import numpy as np
from hparams import hparams as hp
from utils import make_divisible
from torch.utils.data import Dataset


class SpeechDataset(Dataset):
    def __init__(self, path, sort=True):
        super(SpeechDataset, self).__init__()
        self.path = path
        self.meta_data = pd.read_csv(f'{path}/metadata.csv', sep='|',
                                     names=['id', 'frame_nums', 'transcript', 'text'],
                                     usecols=['id', 'frame_nums', 'text'], index_col=False)
        self.meta_data.dropna(inplace=True)

        if sort:
            self.meta_data.sort_values(by=['frame_nums'], inplace=True)

    def __getitem__(self, index):
        id = self.meta_data.iloc[index]['id']
        text = self.meta_data.iloc[index]['text']
        mels = np.load(f'{self.path}/mels/{id}')
        linears = np.load(f'{self.path}/linears/{id}')
        return text, (mels, linears)

    def __len__(self):
        return len(self.meta_data)

def collate_fn(batch):
    inputs = [item[0] for item in batch]
    (mels_target, linears_target) = [item[1] for item in batch]

    input_lens = [len(input) for input in inputs]
    mel_target_lens = [len(mel_target) for mel_target in mels_target]

    max_input_len = max(input_lens)
    max_target_len =  max(mel_target_lens)

    input_batch = np.stack(_pad_input(input, max_input_len) for input in inputs)
    mel_target_batch = np.stack(_pad_target(mel_target, max_target_len) for mel_target in mels_target)
    linear_target_batch = np.stack(_pad_target(linear_target, max_target_len) for linear_target in linears_target)
    stop_token_batch = np.stack(_gen_stop_token(mel_target, max_target_len) for mel_target in mels_target)
    return torch.LongTensor(input_batch), torch.FloatTensor(mel_target_batch), \
            torch.FloatTensor(linear_target_batch), torch.FloatTensor(stop_token_batch)


def _pad_input(input, max_input_len):
    return np.pad(input, (0, max_input_len - len(input)), mode='constant', constant_values=hp.input_padding)

def _pad_target(target, max_target_len):
    max_target_len = make_divisible(max_target_len, hp.outputs_per_step)
    padded = np.zeros(max_target_len - len(target), hp.num_mels) + hp.target_padding
    return np.concatenate((target, padded), axis=0).reshape(max_target_len / hp.outputs_per_step,
                                                            hp.num_mels * hp.outputs_per_step)

def _gen_stop_token(target, max_target_len):
    stop_token = np.zeros(max_target_len)
    stop_token[len(target):] = hp.stop_token_padding
    return stop_token
