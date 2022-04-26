from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import argparse
import random
import copy

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset

from config import BertConfig
from bert_lightning import LitPretrain
from lightning_data_module import EHRDataModule

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)


def random_word(tokens, vocab):
    for i, _ in enumerate(tokens):
        prob = random.random()
        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = "[MASK]"
            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(vocab.word2idx.items()))[0]
            else:
                pass
        else:
            pass

    return tokens


class EHRTokenizer(object):
    """Runs end-to-end tokenization"""

    def __init__(self, data_dir, special_tokens=("[PAD]", "[CLS]", "[MASK]")):

        self.vocab = Voc()

        # special tokens
        self.vocab.add_sentence(special_tokens)

        self.rx_voc = self.add_vocab(os.path.join(data_dir, 'rx-vocab.txt'))
        self.dx_voc = self.add_vocab(os.path.join(data_dir, 'dx-vocab.txt'))
        # code only in multi-visit data
        self.rx_voc_multi = Voc()
        self.dx_voc_multi = Voc()
        with open(os.path.join(data_dir, 'rx-vocab-multi.txt'), 'r') as fin:
            for code in fin:
                self.rx_voc_multi.add_sentence([code.rstrip('\n')])
        with open(os.path.join(data_dir, 'dx-vocab-multi.txt'), 'r') as fin:
            for code in fin:
                self.dx_voc_multi.add_sentence([code.rstrip('\n')])

    def add_vocab(self, vocab_file):
        voc = self.vocab
        specific_voc = Voc()
        with open(vocab_file, 'r') as fin:
            for code in fin:
                voc.add_sentence([code.rstrip('\n')])
                specific_voc.add_sentence([code.rstrip('\n')])
        return specific_voc

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(self.vocab.word2idx[token])
        return ids

    def convert_ids_to_tokens(self, ids):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for i in ids:
            tokens.append(self.vocab.idx2word[i])
        return tokens


class EHRDataset(Dataset):
    def __init__(self, data_pd, tokenizer: EHRTokenizer, max_seq_len):
        self.data_pd = data_pd
        self.tokenizer = tokenizer
        self.seq_len = max_seq_len

        self.sample_counter = 0

        def transform_data(data):
            """
            :param data: raw data form
            :return: {subject_id, [adm, 2, codes]},
            """
            admissions = []
            for _, row in data.iterrows():
                admission = [list(row['ICD9_CODE']), list(row['ATC4'])]
                admissions.append(admission)
            return admissions

        self.admissions = transform_data(data_pd)

    def __len__(self):
        return len(self.admissions)

    def __getitem__(self, item):
        cur_id = item
        adm = copy.deepcopy(self.admissions[item])

        def fill_to_max(l, seq):
            while len(l) < seq:
                l.append('[PAD]')
            return l
        """y
        """
        y_dx = np.zeros(len(self.tokenizer.dx_voc.word2idx))
        y_rx = np.zeros(len(self.tokenizer.rx_voc.word2idx))
        for item in adm[0]:
            y_dx[self.tokenizer.dx_voc.word2idx[item]] = 1
        for item in adm[1]:
            y_rx[self.tokenizer.rx_voc.word2idx[item]] = 1

        """replace tokens with [MASK]
        """
        adm[0] = random_word(adm[0], self.tokenizer.rx_voc)
        adm[1] = random_word(adm[1], self.tokenizer.dx_voc)

        """extract input and output tokens
        """
        random_word
        input_tokens = []  # (2*max_len)
        input_tokens.extend(
            ['[CLS]'] + fill_to_max(list(adm[0]), self.seq_len - 1))
        input_tokens.extend(
            ['[CLS]'] + fill_to_max(list(adm[1]), self.seq_len - 1))

        """convert tokens to id
        """
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

        if cur_id < 5 and False:
            logger.info("*** Example ***")
            logger.info("input tokens: %s" % " ".join(
                [str(x) for x in input_tokens]))
            logger.info("input_ids: %s" %
                        " ".join([str(x) for x in input_ids]))

        cur_tensors = (torch.tensor(input_ids, dtype=torch.long).view(-1, self.seq_len),
                       torch.tensor(y_dx, dtype=torch.float),
                       torch.tensor(y_rx, dtype=torch.float))

        return cur_tensors


def load_dataset(args):
    data_dir = args.data_dir
    max_seq_len = args.max_seq_length

    # load tokenizer
    tokenizer = EHRTokenizer(data_dir)

    # load data
    data = pd.read_pickle(os.path.join(data_dir, 'data-multi-visit.pkl'))
    data_multi = data.iloc[:, :4]
    data_single = pd.read_pickle(
        os.path.join(data_dir, 'data-single-visit.pkl'))

    # load trian, eval, test data
    ids_file = [os.path.join(data_dir, 'train-id.txt'),
                os.path.join(data_dir, 'eval-id.txt'),
                os.path.join(data_dir, 'test-id.txt')]

    def load_ids(data, file_name):
        """
        :param data: multi-visit data
        :param file_name:
        :return: raw data form
        """
        ids = []
        with open(file_name, 'r') as f:
            for line in f:
                ids.append(int(line.rstrip('\n')))
        return data[data['SUBJECT_ID'].isin(ids)].reset_index(drop=True)
    # return tokenizer, \
    #     EHRDataset(load_ids(data_multi, ids_file[1]), tokenizer, max_seq_len), \
    #     EHRDataset(load_ids(data_multi, ids_file[1]), tokenizer, max_seq_len), \
    #     EHRDataset(load_ids(data_multi, ids_file[2]), tokenizer, max_seq_len)
    return tokenizer, \
           EHRDataset(pd.concat([data_single, load_ids(
               data_multi, ids_file[0])]), tokenizer, max_seq_len), \
           EHRDataset(load_ids(data_multi, ids_file[1]), tokenizer, max_seq_len), \
           EHRDataset(load_ids(      data, ids_file[2]), tokenizer, max_seq_len)

def parse_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_name", default='GBert-pretraining', type=str, required=False,
                        help="model name")
    parser.add_argument("--data_dir",
                        default='../data',
                        type=str,
                        required=False,
                        help="The input data dir.")
    parser.add_argument("--pretrain_dir", default='../saved/GBert-predict', type=str, required=False,
                        help="pretraining model dir.")
    parser.add_argument("--train_file", default='data-multi-visit.pkl', type=str, required=False,
                        help="training data file.")
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
    parser.add_argument("--therhold",
                        default=0.3,
                        type=float,
                        help="therhold.")
    parser.add_argument("--max_seq_length",
                        default=55,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
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
    parser.add_argument("--batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=5e-4,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=10.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=1203,
                        help="random seed for initialization")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")

    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.model_name)
    return args

if __name__ == '__main__':
    args = parse_arguments()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading Dataset")
    ehr_data_pretrain = EHRDataModule(is_pretrain=True, data_dir='../data', batch_size=64)
    ehr_data_train = EHRDataModule(is_pretrain=False, data_dir='../data', batch_size=64)
    ehr_data_pretrain.setup()
    ehr_data_train.setup()
    tokenizer = ehr_data_pretrain.tokenizer
    train_dataloader = ehr_data_pretrain.train_dataloader()
    val_dataloader = ehr_data_pretrain.val_dataloader()

    # TODO: Load the model here

    config = BertConfig(
        vocab_size_or_config_json_file=len(tokenizer.vocab.word2idx))
    config.graph = args.graph
    model = LitPretrain(config, tokenizer, args.learning_rate)

    # TODO: Save the model here

    trainer = pl.Trainer(
        accelerator='cpu' if args.no_cuda else 'gpu',
        max_epochs=args.num_train_epochs,
        default_root_dir='../saved/lightning')
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model, dataloaders=test_dataloader)
    model.logger.save()

    print('done')
