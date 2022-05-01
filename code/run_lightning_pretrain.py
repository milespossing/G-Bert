import argparse
import os

import pytorch_lightning as pl
from loggers import ScreenLogger
from pytorch_lightning.loggers import TensorBoardLogger, LoggerCollection
from pytorch_lightning.callbacks import ModelCheckpoint

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
                        required=False,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--checkpoint",
                        help="checkpoint file (*.ckpt) to load from",
                        action='store_true',
                        default=False,
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
    parser.add_argument("--pretrain_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for pretrain task.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)
    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading Dataset")
    ehr_data = EHRDataModule(is_pretrain=True, data_dir='../data', batch_size=args.pretrain_batch_size)
    ehr_data_test = EHRDataModule(is_pretrain=False, data_dir='../data', batch_size=args.pretrain_batch_size)
    ehr_data.setup()
    ehr_data_test.setup()
    tokenizer = ehr_data.tokenizer

    config = BertConfig(
        vocab_size_or_config_json_file=len(tokenizer.vocab.word2idx))
    config.graph = args.graph

    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(args.output_dir, 'checkpoints'),
                                          save_top_k=1,
                                          monitor='val_rx2rx_prauc',
                                          mode='max',
                                          filename='pretrain',
                                          save_weights_only=True,
                                          save_last=True)

    tb_logger = TensorBoardLogger(args.output_dir, name='pretrain')
    screen = ScreenLogger()

    trainer = pl.Trainer(
        accelerator='cpu' if args.no_cuda else 'gpu',
        max_epochs=args.num_train_epochs,
        logger=LoggerCollection([tb_logger, screen]),
        callbacks=[checkpoint_callback],
        default_root_dir=args.output_dir)

    model = LitGBert(config, tokenizer, ehr_data_test.tokenizer, args.learning_rate, args.threshold)
    model.set_mode(BertMode.Pretrain)
    trainer.fit(model, ehr_data)
    final_model = LitGBert.load_from_checkpoint(os.path.join(args.output_dir, 'checkpoints', 'pretrain.ckpt'),
                                                config=config,
                                                tokenizer_pretrain=tokenizer, tokenizer_predict=ehr_data_test.tokenizer,
                                                learning_rate=args.learning_rate, eval_threshold=args.threshold)
    final_model.set_mode(BertMode.Pretrain)
    trainer.test(final_model, ehr_data)

    model.logger.save()
