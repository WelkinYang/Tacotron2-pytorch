import os
import sys
import logging
import argparse
from model.model_utils import create_model
from hparams import hparams as hp

import torch
import torch.nn as nn

def train(args, exp_name, step):
    pass

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
    parser.add_argument('--epochs', default=5000000, type=int,
                        help='number of epochs to train a tacotron2')
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

    train(args, model, exp_name, step)

if __name__ == '__main__':
    main()