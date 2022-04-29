import argparse
import random
import os

import numpy as np
import torch
import pytorch_lightning as pl

from lightning_data_module import EHRDataModule
from config import BertConfig
from bert_lightning import LitBert
from run_gbert import load_dataset
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',
                        type=int,
                        default=1203,
                        help="random seed for initialization")
    parser.add_argument("--graph",
                        default=False,
                        action='store_true',
                        help="if use ontology embedding")
    parser.add_argument("--output_dir",
                        default='../saved/',
                        type=str,
                        required=False,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--checkpoint",
                        help="checkpoint file (*.ckpt) to load from",
                        type=str,
                        required=False)
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--num_train_epochs",
                        default=10.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--learning_rate",
                        default=5e-4,
                        type=float,
                        help="The initial learning rate for Adam.")
    return parser.parse_args()
    pass


if __name__ == '__main__':
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading Dataset")
    ehr_data = EHRDataModule(is_pretrain=True, data_dir='../data', batch_size=64)
    ehr_data_test = EHRDataModule(is_pretrain=False, data_dir='../data', batch_size=64)
    ehr_data.setup()
    ehr_data_test.setup()
    tokenizer = ehr_data.tokenizer

    config = BertConfig(
        vocab_size_or_config_json_file=len(tokenizer.vocab.word2idx))
    config.graph = args.graph
    model = LitBert(config, tokenizer, ehr_data_test.tokenizer, args.learning_rate)

    trainer = pl.Trainer(
        accelerator='cpu' if args.no_cuda else 'gpu',
        max_epochs=args.num_train_epochs,
        default_root_dir=args.output_dir)
    model.use_pretrain()

    if args.checkpoint is not None:
        trainer.fit(model, ehr_data, ckpt_path=os.path.join(args.output_dir, 'checkpoint.ckpt'))
    else:
        trainer.fit(model, ehr_data)
        trainer.save_checkpoint(os.path.join(args.output_dir, 'checkpoint.ckpt'))

    model.logger.save()
    model.use_predict()
    trainer.fit(model, ehr_data_test)

    # TODO: need to use a conditional saving mechanism like in run_pretraining.py:457
    #       this could be stored by the val_epoch_end routine

    print('testing...')

    trainer.test(model, ehr_data_test)

    with open(os.path.join(args.output_dir, 'bert_config.json'), 'w', encoding='utf-8') as fout:
        fout.write(model.config.to_json_string())

    print('done')
