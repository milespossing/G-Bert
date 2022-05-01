from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import argparse
import pytorch_lightning as pl
from loggers import ScreenLogger
from pytorch_lightning.loggers import TensorBoardLogger, LoggerCollection
from pytorch_lightning.callbacks import ModelCheckpoint
from config import BertConfig
from lightning_model import LitGBert, BertMode
from lightning_data_module import EHRDataModule
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_name", default='GBert-lightning', type=str, required=False,
                        help="model name")
    parser.add_argument("--enable_progress_bar",
                        default='True',
                        type=str,
                        choices=('True', 'False'))
    parser.add_argument("--data_dir",
                        default='../data',
                        type=str,
                        required=False,
                        help="The input data dir.")
    parser.add_argument("--pretrain_dir", default='../saved/GBert-lightning', type=str, required=False,
                        help="pretraining model dir.")
    parser.add_argument("--output_dir",
                        default='../saved/',
                        type=str,
                        required=False,
                        help="The output directory where the model checkpoints will be written.")

    # Other parameters
    parser.add_argument("--use_pretrain",
                        default=False,
                        action='store_true',
                        help="if use ontology embedding")
    parser.add_argument("--graph",
                        default=False,
                        action='store_true',
                        help="if use ontology embedding")
    parser.add_argument("--threshold",
                        default=0.3,
                        type=float,
                        help="threshold for metrics eval.")
    parser.add_argument("--max_seq_length",
                        default=55,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--no_pretrain",
                        default=False,
                        action='store_true',
                        help="Don't do the pretrain step")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        default=True,
                        action='store_true',
                        help="Whether to run on the dev set.")
    parser.add_argument("--do_test",
                        default=True,
                        action='store_true',
                        help="Whether to run on the test set.")
    parser.add_argument("--pretrain_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for pretrain task.")
    parser.add_argument("--learning_rate",
                        default=5e-4,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=5,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--num_train_repeats",
                        default=1,
                        type=int,
                        help="Total number of pretrain + predict train repeats.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=1203,
                        help="random seed for initialization")

    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.model_name)
    return args


if __name__ == '__main__':
    args = parse_arguments()
    pl.seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("Loading Dataset")
    ehr_data_pretrain = EHRDataModule(is_pretrain=True, data_dir='../data', batch_size=args.pretrain_batch_size)
    ehr_data_predict = EHRDataModule(is_pretrain=False, data_dir='../data', batch_size=1)
    ehr_data_pretrain.setup()
    ehr_data_predict.setup()
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(args.output_dir, 'checkpoints'),
                                          save_top_k=2,
                                          save_last=True,
                                          monitor='val_rx2rx_prauc')

    assert ehr_data_pretrain.tokenizer.vocab.word2idx == ehr_data_predict.tokenizer.vocab.word2idx, \
        'Pretrain and predict data-loaders are not using the same vocabulary!'

    # todo: set the hyperparameter search here
    config = BertConfig(len(ehr_data_pretrain.tokenizer.vocab.word2idx), graph=args.graph)

    # todo 'if args.use_pretrain:'
    model = LitGBert(config, ehr_data_pretrain.tokenizer, ehr_data_predict.tokenizer, args.learning_rate,
                     args.threshold)
    tb_logger_pretrain = TensorBoardLogger(args.output_dir, name='pretrain')
    tb_logger_predict = TensorBoardLogger(args.output_dir, name='predict')
    screen = ScreenLogger()

    # TODO: Save the model here
    for i_tandem_train in range(args.num_train_repeats):
        trainer = pl.Trainer(
            accelerator='cpu' if args.no_cuda else 'gpu',
            max_epochs=args.num_train_epochs,
            logger=LoggerCollection([tb_logger_pretrain, screen]),
            default_root_dir='../saved/lightning',
            enable_progress_bar=args.enable_progress_bar == 'True')
        if not args.no_pretrain:
            model.set_mode(BertMode.Pretrain)
            trainer.fit(model, ehr_data_pretrain)
            trainer.test(model, ehr_data_pretrain)
            trainer.logger.save()
        else:
            logger.info('No pretrain')
        model.set_mode(BertMode.Predict)
        trainer = pl.Trainer(
            accelerator='cpu' if args.no_cuda else 'gpu',
            max_epochs=args.num_train_epochs,
            logger=LoggerCollection([tb_logger_predict, screen]),
            default_root_dir='../saved/lightning',
            enable_progress_bar=args.enable_progress_bar == 'True')
        if args.do_train:
            trainer.fit(model, ehr_data_predict)
        trainer.test(model, ehr_data_predict)
        trainer.logger.save()

    print('done')
