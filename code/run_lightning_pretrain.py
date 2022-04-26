import argparse
import random
import os

import numpy as np
import torch
import pytorch_lightning as pl

from lightning_data_module import EHRDataModule
from config import BertConfig
from bert_lightning import LitPretrain


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
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading Dataset")
    ehr_data = EHRDataModule(is_pretrain=True, data_dir='../data', batch_size=64)
    ehr_data.setup()
    tokenizer = ehr_data.tokenizer
    train_dataloader = ehr_data.train_dataloader(num_workers=12)
    val_dataloader = ehr_data.val_dataloader(num_workers=12)

    config = BertConfig(
        vocab_size_or_config_json_file=len(tokenizer.vocab.word2idx))
    config.graph = args.graph
    model = LitPretrain(config, tokenizer, args.learning_rate)
    trainer = pl.Trainer(
        accelerator='cpu' if args.no_cuda else 'gpu',
        max_epochs=args.num_train_epochs,
        default_root_dir=args.output_dir)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    model.logger.save()

    # TODO: need to use a conditional saving mechanism like in run_pretraining.py:457
    #       this could be stored by the val_epoch_end routine
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'model.bin'))
    with open(os.path.join(args.output_dir, 'bert_config.json'), 'w', encoding='utf-8') as fout:
        fout.write(model.config.to_json_string())

    print('done')
