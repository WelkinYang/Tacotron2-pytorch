import os
import sys
import tqdm
import logging
import argparse
from tensorboardX import SummaryWriter

from hparams import hparams as hp
from model.model_utils import create_model
from datasets import SpeechDataset, collate_fn


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler


def train(args, model, exp_name, step, device):
    data_path = os.path.join(args.base_dir, args.data)
    dataset = SpeechDataset(data_path)

    optimizer = optim.Adam(model.parameters(), lr=hp.inital_learning_rate, weight_decay=hp.decay_rate)

    sampler = RandomSampler(dataset)
    loader = DataLoader(dataset, batch_size=hp.batch_size, sampler=sampler, num_workers=6, collate_fn=collate_fn, pin_memory=True)

    while True:
        pbar = tqdm(loader, total=len(loader), unit=' batches')
        for b, (input_batch, mel_target_batch, linear_target_batch, stop_token_batch) in enumerate(pbar):

            input = input_batch.to(device=device)
            mel_target = mel_target_batch.to(device=device)
            linear_target = linear_target_batch.to(device=device)
            stop_token_target = stop_token_batch.to(device=device)

            decoder_outputs, mel_outputs, linear_outputs, stop_token_prediction = \
                model(input, mel_target, linear_target, stop_token_target)

            decoder_loss = F.mse_loss(decoder_outputs, mel_target)
            mel_loss = F.mse_loss(mel_outputs, mel_target)

            loss = decoder_loss + mel_loss

            if hp.use_linear_spec:
                linear_loss = F.mse_loss(linear_outputs, linear_target)
                loss += linear_loss

            if hp.use_stop_token:
                stop_token_loss = F.binary_cross_entropy(stop_token_prediction, stop_token_target, reduction='sum')
                loss += stop_token_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step = step + 1
            logging.info(f'loss: {loss.item():.4f} at step{step}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='')
    parser.add_argument('--data', default='./training_data/', type=str,
                        help='path to dataset contains inputs and targets')
    parser.add_argument('--log_dir', default='tacotron2', type=str, help='path to save alignment and checkpoint')
    parser.add_argument('--restore_from', default=None, type=str,
                        help='the checkpoint restored from the log_dir you set')
    parser.add_argument('--summary_interval', type=int, default=250,
                        help='Steps between running summary ops')
    parser.add_argument('--checkpoint_interval', type=int, default=2500,
                        help='Steps between writing checkpoints')
    parser.add_argument('--eval_interval', type=int, default=5000,
                        help='Steps between eval on test data')
    parser.add_argument('--name', default='tacotron2', type=str,
                        help='name of the experiment')

    args = parser.parse_args()
    logging.basicConfig(format='%(asctime)s %(filename)s %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S', level=logging.DEBUG,
                        stream=sys.stdout)

    step = 0
    exp_name = args.name
    model = create_model()
    if args.restore_from is not None:
        checkpoint_path = os.path.join(args.log_dir, os.path.join("checkpoint", args.restore_from))
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path))
            step = int(checkpoint_path.split('/')[-1].split('_')[-1].split(".")[0])
            exp_name = checkpoint_path.split('/')[-1].split("_")[0]
        else:
            logging.error(f'checkpoint path:{checkpoint_path} does\'t exist!')

    os.environ["CUDA_VISIBEL_DEVICES"] = hp.gpu_ids
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    train(args, model, exp_name, step, device)

if __name__ == '__main__':
    main()