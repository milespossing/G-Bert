from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Union
from pytorch_lightning.trainer.states import RunningStage
from lightning_model import LitGBert, BertMode, EHRDataModule
from argparse import ArgumentParser, Namespace
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl


class ExperimentManager(Callback):
    def __init__(self, args: Union[Namespace, ArgumentParser]):
        self.model = LitGBert(args)
        if 0 < args.toggle_mode_every_n_steps < args.max_epochs:
            self.toggle_mode_every_n_steps = args.toggle_mode_every_n_steps
            self.model.set_mode(BertMode.Pretrain)
        else:
            self.toggle_mode_every_n_steps = None
            self.model.set_mode(BertMode.Predict)
        self._epoch_counter = 0

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._epoch_counter += 1
        if self.toggle_mode_every_n_steps and (self._epoch_counter % self.toggle_mode_every_n_steps == 0):
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

    @staticmethod
    def parse_general_arguments(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument("--exp_name",
                            default='my_experiment',
                            type=str,
                            required=True,
                            help="The experiment name for logging.")
        parser.add_argument("--data_dir",
                            default='../data',
                            type=str,
                            required=False,
                            help="The input data dir.")
        parser.add_argument("--output_dir",
                            default='../saved/',
                            type=str,
                            required=False,
                            help="The output directory where the model checkpoints and logs will be written.")
        parser.add_argument('--seed',
                            type=int,
                            default=1203,
                            help="random seed for initialization")
        parser.add_argument("--no_cuda",
                            action='store_true',
                            help="Whether not to use CUDA when available")
        parser.add_argument("-trn", "--do_train",
                            action='store_true',
                            help="Whether to run training.")
        parser.add_argument("-evl", "--do_eval",
                            action='store_true',
                            help="Whether to run on the val set.")
        parser.add_argument("-tb", "--do_test_pretrain",
                            action='store_true',
                            help="Whether to run on the test set on pretrain task.")
        parser.add_argument("-tp", "--do_test_predict",
                            action='store_true',
                            help="Whether to run on the test set on predict task.")
        parser.add_argument("-tg_int", "--toggle_mode_every_n_steps",
                            default=5,
                            type=int,
                            help="Toggle interval between pretrain/predict tasks.")
        return parser


def collect_all_args(configuration_string: str) -> Namespace:
    parser = ArgumentParser(description="G-Bert training session")
    # let the model add what it wants
    parser = ExperimentManager.parse_general_arguments(parser)
    parser = Trainer.add_argparse_args(parser)
    parser = LitGBert.add_model_specific_args(parser)
    parser = EHRDataModule.add_model_specific_args(parser)
    return parser.parse_args(configuration_string.strip('\n').split(' '))


def main(hparams: Namespace):
    dict_args = vars(hparams)
    seed_everything(dict_args['seed'], workers=True)
    exp_mnr = ExperimentManager(hparams)
    tb_logger = TensorBoardLogger(dict_args['output_dir'], dict_args['exp_name'], default_hp_metric=False)
    trainer = Trainer.from_argparse_args(hparams, **{'logger': tb_logger, 'callbacks': [exp_mnr]})
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
