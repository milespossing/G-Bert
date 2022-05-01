import os
import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, LoggerCollection

from loggers import ScreenLogger
from lightning_data_module import EHRDataModule
from config import BertConfig
from lightning_model import LitGBert, BertMode


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',
                        type=int,
                        default=1203,
                        help="random seed for initialization")
    parser.add_argument("--threshold",
                        default=0.3,
                        type=float,
                        help="threshold for metrics eval.")
    parser.add_argument("--graph",
                        default=False,
                        action='store_true',
                        help="if use ontology embedding")
    parser.add_argument("--output_dir",
                        default='../saved/',
                        type=str,
                        required=False)
    parser.add_argument('--do_train',
                        default=False,
                        action='store_true',
                        help='Run training step')
    parser.add_argument("--checkpoint_dir",
                        type=str,
                        required=False,
                        help="The directory where the checkpoints have been written.")
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
    parser.add_argument("--pretrain_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for pretrain task.")
    return parser.parse_args()


def main(args):
    ehr_data = EHRDataModule(is_pretrain=True, data_dir='../data', batch_size=args.pretrain_batch_size)
    ehr_data_test = EHRDataModule(is_pretrain=False, data_dir='../data', batch_size=args.pretrain_batch_size)
    ehr_data.setup()
    ehr_data_test.setup()
    config = BertConfig(vocab_size_or_config_json_file=len(ehr_data.tokenizer.vocab.word2idx))
    config.graph = args.graph
    tb_logger = TensorBoardLogger(args.output_dir, name='predict')
    screen = ScreenLogger()
    trainer = pl.Trainer(
        accelerator='cpu' if args.no_cuda else 'gpu',
        max_epochs=args.num_train_epochs,
        logger=LoggerCollection([tb_logger, screen]),
        default_root_dir=args.output_dir)
    if args.checkpoint_dir is not None:
        print('loading from ' + args.checkpoint_dir)
        model = LitGBert.load_from_checkpoint(os.path.join(args.checkpoint_dir, 'pretrain.ckpt'),
                                              config=config, tokenizer_pretrain=ehr_data.tokenizer,
                                              tokenizer_predict=ehr_data_test.tokenizer,
                                              learning_rate=args.learning_rate, eval_threshold=args.threshold)
    else:
        print('training from scratch')
        model = LitGBert(config, ehr_data.tokenizer, ehr_data_test.tokenizer, args.learning_rate, args.threshold)
    model.set_mode(BertMode.Predict)
    if args.do_train:
        trainer.fit(model, ehr_data_test)
    else:
        print('skipping training')
    trainer.test(model, ehr_data_test)


if __name__ == '__main__':
    run_args = parse_args()
    pl.seed_everything(run_args.seed, workers=True)
    main(run_args)
