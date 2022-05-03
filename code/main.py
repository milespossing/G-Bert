from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
from typing import Union

from pytorch_lightning.trainer.states import RunningStage

from config import BertConfig
from lightning_model import LitGBert, BertMode, EHRDataModule
from argparse import ArgumentParser, Namespace
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback
from tensorboard import program as tb_program
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from collections import defaultdict


class ExperimentManager(Callback):
    def __init__(self, args: Union[Namespace, ArgumentParser]):
        self.model = LitGBert(args)
        self.model.set_mode(BertMode.Pretrain)
        self.toggle_mode_every_n_steps = args.toggle_mode_every_n_steps
        self._epoch_counter = 0

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._epoch_counter += 1
        if self.toggle_mode_every_n_steps > 0 and (self._epoch_counter % self.toggle_mode_every_n_steps == 0):
            # trainer.test(self.model, self.model.ehr_data)
            if self.model.mode == BertMode.Pretrain:
                self.model.set_mode(BertMode.Predict)
            else:
                self.model.set_mode(BertMode.Pretrain)
            self.model.ehr_data.setup(RunningStage.TRAINING)
            trainer.reset_train_dataloader()
            trainer.reset_train_val_dataloaders()
            trainer.reset_val_dataloader()

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.model.ehr_data.setup(RunningStage.TESTING)
        trainer.reset_test_dataloader()


def main(hparams):
    dict_args = vars(hparams)
    seed_everything(dict_args['seed'], workers=True)
    exp_mnr = ExperimentManager(hparams)
    tb_logger = TensorBoardLogger(dict_args['output_dir'], name="gbert_05_03")
    hparams.logger = tb_logger
    hparams.callbacks = [exp_mnr]
    trainer = Trainer.from_argparse_args(hparams)
    if dict_args['do_train']:
        if dict_args['do_eval']:
            trainer.fit(exp_mnr.model, exp_mnr.model.ehr_data)
        else:
            trainer.fit(exp_mnr.model, exp_mnr.model.ehr_data.train_dataloader())
    if dict_args['do_test_pretrain']:
        exp_mnr.model.set_mode(BertMode.Pretrain)
        trainer.test(exp_mnr.model, exp_mnr.model.ehr_data)
    if dict_args['do_test_predict']:
        exp_mnr.model.set_mode(BertMode.Predict)
        trainer.test(exp_mnr.model, exp_mnr.model.ehr_data)


def parse_general_arguments(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--data_dir",
                        default='../data',
                        type=str,
                        required=False,
                        help="The input data dir.")
    parser.add_argument("--output_dir",
                        default='../saved/',
                        type=str,
                        required=False,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--pretrain_dir", default='../saved/GBert-lightning', type=str, required=False,
                        help="pretraining model dir.")
    parser.add_argument('--seed',
                        type=int,
                        default=1203,
                        help="random seed for initialization")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run on the dev set.")
    parser.add_argument("--do_test_pretrain",
                        action='store_true',
                        help="Whether to run on the test set.")
    parser.add_argument("--do_test_predict",
                        action='store_true',
                        help="Whether to run on the test set.")
    parser.add_argument("--toggle_mode_every_n_steps",
                        default=5,
                        type=int,
                        help="Total number of pretrain + predict train repeats.")
    return parser


if __name__ == "__main__":
    with open('batch_config.txt') as f:
        runs_config = f.readlines()
    last_output_dir = None
    for i_config, config_str in enumerate(runs_config):
        parser = ArgumentParser(description="G-Bert training session")
        parser = parse_general_arguments(parser)
        parser = Trainer.add_argparse_args(parser)

        # let the model add what it wants
        parser = LitGBert.add_model_specific_args(parser)
        parser = EHRDataModule.add_model_specific_args(parser)

        args = parser.parse_args(config_str.split(' '))
        main(args)
        last_output_dir = args.output_dir

    tb = tb_program.TensorBoard()
    tb.configure(argv=[None, '--logdir', last_output_dir])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")
    input("Press Enter to continue...")

