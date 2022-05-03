from typing import Dict, Tuple, Union
import enum

from pytorch_lightning.trainer.states import RunningStage
from torch import nn
from torch.nn import functional as F

from config import BertConfig
from bert_models import FuseEmbeddings, BertEmbeddings, TransformerBlock, LayerNorm, BERT
from predictive_models import SelfSupervisedHead, MappingHead
from utils import t2n, multi_label_metric
import os
from argparse import ArgumentParser, Namespace
from typing import Optional, Dict, List
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset  # random_split,
import random
from os.path import join
import copy
import numpy as np
import torch


class Voc(object):
    def __init__(self):
        self.idx2word = {}
        self.word2idx = {}

    def add_sentence(self, sentence):
        for word in sentence:
            if word not in self.word2idx:
                self.idx2word[len(self.word2idx)] = word
                self.word2idx[word] = len(self.word2idx)


class Tokenizer(object):
    """Runs end-to-end tokenization"""

    def __init__(self, data_dir, is_pretrain: bool, special_tokens=("[PAD]", "[CLS]", "[MASK]")):

        self.vocab = Voc()

        # special tokens
        self.vocab.add_sentence(special_tokens)

        self.rx_voc = self.add_vocab(join(data_dir, 'rx-vocab.txt'))
        self.dx_voc = self.add_vocab(join(data_dir, 'dx-vocab.txt'))

        if not is_pretrain:
            # code only in multi-visit data
            self.rx_voc_multi = Voc()
            self.dx_voc_multi = Voc()
            with open(join(data_dir, 'rx-vocab-multi.txt'), 'r') as fin:
                for code in fin:
                    self.rx_voc_multi.add_sentence([code.rstrip('\n')])
            with open(join(data_dir, 'dx-vocab-multi.txt'), 'r') as fin:
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


class EHRDatasetTemplate(Dataset):
    def __init__(self, data_pd, tokenizer: Tokenizer, max_seq_len):
        self.data_pd = data_pd
        self.tokenizer = tokenizer
        self.seq_len = max_seq_len
        self.data = self.__transform_data__(data_pd)

    @classmethod
    def __transform_data__(cls, data: pd.DataFrame):
        raise NotImplementedError('transform_data() not implemented')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        raise NotImplementedError('__getitem__() not implemented')


class DatasetPretrain(EHRDatasetTemplate):
    # mask token with 15% probability
    mask_prob = 0.15
    # 80% randomly change token to mask token
    change_to_mask_prob = 0.8
    # 10% randomly change token to random token
    change_to_random_prob = 0.9

    def __init__(self, data_pd, tokenizer: Tokenizer, max_seq_len):
        super(DatasetPretrain, self).__init__(data_pd, tokenizer, max_seq_len)

    @classmethod
    def __transform_data__(cls, data) -> List[List[List]]:  # todo figure out return type
        """
        :param data: raw data form
        :return: {subject_id, [adm, 2, codes]},
        """
        admissions = []
        for _, row in data.iterrows():
            admission = [list(row['ICD9_CODE']), list(row['ATC4'])]
            admissions.append(admission)
        return admissions

    def __getitem__(self, item):
        adm = copy.deepcopy(self.data[item])

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
        adm[0] = self.__random_word__(adm[0], self.tokenizer.rx_voc)
        adm[1] = self.__random_word__(adm[1], self.tokenizer.dx_voc)

        """extract input and output tokens
        """
        # random_word  # todo: figure out
        input_tokens = []  # (2*max_len)
        input_tokens.extend(
            ['[CLS]'] + fill_to_max(list(adm[0]), self.seq_len - 1))
        input_tokens.extend(
            ['[CLS]'] + fill_to_max(list(adm[1]), self.seq_len - 1))

        """convert tokens to id
        """
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)

        # todo: decide about logging
        # if cur_id < 5:
        #     logger.info("*** Example ***")
        #     logger.info("input tokens: %s" % " ".join(
        #         [str(x) for x in input_tokens]))
        #     logger.info("input_ids: %s" %
        #                 " ".join([str(x) for x in input_ids]))

        cur_tensors = (torch.tensor(input_ids, dtype=torch.long).view(-1, self.seq_len),
                       torch.tensor(y_dx, dtype=torch.float),
                       torch.tensor(y_rx, dtype=torch.float))

        return cur_tensors

    def __random_word__(self, tokens, vocab):
        for i, _ in enumerate(tokens):
            prob = random.random()
            # mask token with 15% probability
            if prob < self.mask_prob:
                prob /= self.mask_prob

                # 80% randomly change token to mask token
                if prob < self.change_to_mask_prob:
                    tokens[i] = "[MASK]"
                # 10% randomly change token to random token
                elif prob < self.change_to_random_prob:
                    tokens[i] = random.choice(list(vocab.word2idx.items()))[0]
                else:
                    pass
            else:
                pass
        return tokens


class DatasetPrediction(EHRDatasetTemplate):
    def __init__(self, data_pd, tokenizer: Tokenizer, max_seq_len):
        super(DatasetPrediction, self).__init__(data_pd, tokenizer, max_seq_len)

    @classmethod
    def __transform_data__(cls, data) -> Dict[int, List[List[List]]]:
        """
        :param data: raw data form
        :return: {subject_id, [adm, 2, codes]},
        """
        records = {}
        for subject_id in data['SUBJECT_ID'].unique():
            item_df = data[data['SUBJECT_ID'] == subject_id]
            patient = []
            for _, row in item_df.iterrows():
                admission = [list(row['ICD9_CODE']), list(row['ATC4'])]
                patient.append(admission)
            if len(patient) < 2:
                continue
            records[subject_id] = patient
        return records

    def __getitem__(self, item):
        subject_id = list(self.data.keys())[item]

        def fill_to_max(l, seq):
            while len(l) < seq:
                l.append('[PAD]')
            return l

        """extract input and output tokens
        """
        input_tokens = []  # (2*max_len*adm)
        output_dx_tokens = []  # (adm-1, l)
        output_rx_tokens = []  # (adm-1, l)

        for idx, adm in enumerate(self.data[subject_id]):
            input_tokens.extend(
                ['[CLS]'] + fill_to_max(list(adm[0]), self.seq_len - 1))
            input_tokens.extend(
                ['[CLS]'] + fill_to_max(list(adm[1]), self.seq_len - 1))
            # output_rx_tokens.append(list(adm[1]))

            if idx != 0:
                output_rx_tokens.append(list(adm[1]))
                output_dx_tokens.append(list(adm[0]))

        """convert tokens to id
        """
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        output_dx_labels = []  # (adm-1, dx_voc_size)
        output_rx_labels = []  # (adm-1, rx_voc_size)

        dx_voc_size = len(self.tokenizer.dx_voc_multi.word2idx)
        rx_voc_size = len(self.tokenizer.rx_voc_multi.word2idx)
        for tokens in output_dx_tokens:
            tmp_labels = np.zeros(dx_voc_size)
            tmp_labels[list(
                map(lambda x: self.tokenizer.dx_voc_multi.word2idx[x], tokens))] = 1
            output_dx_labels.append(tmp_labels)

        for tokens in output_rx_tokens:
            tmp_labels = np.zeros(rx_voc_size)
            tmp_labels[list(
                map(lambda x: self.tokenizer.rx_voc_multi.word2idx[x], tokens))] = 1
            output_rx_labels.append(tmp_labels)

        # todo: decide about logging
        # if cur_id < 5:
        #     logger.info("*** Example ***")
        #     logger.info("subject_id: %s" % subject_id)
        #     logger.info("input tokens: %s" % " ".join(
        #         [str(x) for x in input_tokens]))
        #     logger.info("input_ids: %s" %
        #                 " ".join([str(x) for x in input_ids]))

        # todo: move assertion out of the loop
        # assert len(input_ids) == (self.seq_len *
        #                           2 * len(self.data[subject_id]))
        # assert len(output_dx_labels) == (len(self.data[subject_id]) - 1)
        # assert len(output_rx_labels) == len(self.records[subject_id])-1

        cur_tensors = (torch.tensor(input_ids).view(-1, self.seq_len),
                       torch.tensor(output_dx_labels, dtype=torch.float),
                       torch.tensor(output_rx_labels, dtype=torch.float))

        return cur_tensors


class EHRDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data", max_seq_len=55, pretrain_batch_size=10, use_single=True):
        super().__init__()
        self.train_dataset, self.eval_dataset, self.test_dataset = (None, None, None)
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        self.pretrain_batch_size = pretrain_batch_size
        self.use_single = use_single
        # load tokenizer
        self.tokenizer_pretrain = Tokenizer(self.data_dir, True)
        self.tokenizer_predict = Tokenizer(self.data_dir, False)
        assert self.tokenizer_pretrain.vocab.word2idx == self.tokenizer_predict.vocab.word2idx, \
            'Pretrain and predict data-loaders are not using the same vocabulary!'
        self.mode = None

    @staticmethod
    def add_model_specific_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument("--pretrain_batch_size",
                            default=64,
                            type=int,
                            help="Total batch size for pretrain task.")
        parser.add_argument("--max_seq_length",
                            default=55,
                            type=int,
                            help="The maximum total input sequence length after WordPiece tokenization. \n"
                                 "Sequences longer than this will be truncated, and sequences shorter \n"
                                 "than this will be padded.")
        parser.add_argument("--use_single",
                            action='store_true',
                            help="Whether to run on the dev set.")
        return parser

    def prepare_data(self):
        # TODO: here we download & save data (part of load_dataset from orig code)
        pass

    def setup(self, stage: Optional[RunningStage]):
        # TODO: data operations you might want to perform on every GPU (here goes most of orig code)

        # load train, eval, test index from files
        ids_file = [os.path.join(self.data_dir, 'train-id.txt'),
                    os.path.join(self.data_dir, 'eval-id.txt'),
                    os.path.join(self.data_dir, 'test-id.txt')]

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

        assert self.mode is not None
        if self.mode == BertMode.Pretrain:
            data_multi = pd.read_pickle(os.path.join(self.data_dir, 'data-multi-visit.pkl')).iloc[:, :4]
            if self.use_single:
                data_single = pd.read_pickle(os.path.join(self.data_dir, 'data-single-visit.pkl'))
                self.train_dataset = DatasetPretrain(pd.concat([data_single, load_ids(data_multi, ids_file[0])]),
                                                     self.tokenizer_pretrain, self.max_seq_len)
            else:
                self.train_dataset = DatasetPretrain(load_ids(data_multi, ids_file[0]),
                                                     self.tokenizer_pretrain, self.max_seq_len)
            self.eval_dataset = DatasetPretrain(load_ids(data_multi, ids_file[1]),
                                                self.tokenizer_pretrain, self.max_seq_len)
            self.test_dataset = DatasetPretrain(load_ids(data_multi, ids_file[2]),
                                                self.tokenizer_pretrain, self.max_seq_len)
        else:  # Prediction task
            data_multi = pd.read_pickle(os.path.join(self.data_dir, 'data-multi-visit.pkl'))
            self.train_dataset, self.eval_dataset, self.test_dataset = \
                tuple(map(lambda x: DatasetPrediction(load_ids(data_multi, x), self.tokenizer_predict, self.max_seq_len), ids_file))
        pass

    def train_dataloader(self, num_workers=6):
        batch_size = self.pretrain_batch_size if isinstance(self.train_dataset, DatasetPretrain) else 1
        return DataLoader(self.train_dataset,
                          sampler=RandomSampler(self.train_dataset),
                          batch_size=batch_size, num_workers=num_workers)

    def val_dataloader(self, num_workers=1):
        batch_size = self.pretrain_batch_size if isinstance(self.train_dataset, DatasetPretrain) else 1
        return DataLoader(self.eval_dataset,
                          sampler=SequentialSampler(self.eval_dataset),
                          batch_size=batch_size,
                          num_workers=num_workers)

    def test_dataloader(self, num_workers=1):
        batch_size = self.pretrain_batch_size if isinstance(self.train_dataset, DatasetPretrain) else 1
        return DataLoader(self.test_dataset,
                          sampler=SequentialSampler(self.test_dataset),
                          batch_size=batch_size,
                          num_workers=num_workers)

def metric_report(threshold=0.5):
    def get_metrics(y_pred, y_true):
        y_prob = y_pred.copy()
        y_pred[y_pred > threshold] = 1
        y_pred[y_pred <= threshold] = 0
        ja, prauc, avg_p, avg_r, avg_f1 = multi_label_metric(
            y_true, y_pred, y_prob)
        return {'jaccard':ja, 'f1': avg_f1, 'prauc':prauc}

    return lambda pred, true: get_metrics(pred, true)


class BertMode(enum.Enum):
    Pretrain = 0
    Predict = 1


class LitGBert(pl.LightningModule):
    def __init__(self, args: Union[Namespace, ArgumentParser]):
        super().__init__()
        # equivalent
        self.save_hyperparameters(args)
        self.ehr_data = EHRDataModule(args.data_dir, args.max_seq_length, args.pretrain_batch_size, args.use_single)
        config = BertConfig(len(self.ehr_data.tokenizer_pretrain.vocab.word2idx), graph=args.graph)
        assert self.ehr_data.tokenizer_predict.dx_voc.word2idx == self.ehr_data.tokenizer_pretrain.dx_voc.word2idx
        assert self.ehr_data.tokenizer_predict.rx_voc.word2idx == self.ehr_data.tokenizer_pretrain.rx_voc.word2idx
        self.dx_voc_size = len(self.ehr_data.tokenizer_pretrain.dx_voc.word2idx)
        self.rx_voc_size = len(self.ehr_data.tokenizer_pretrain.rx_voc.word2idx)

        # pretrain part
        self.bert = BERT(config, self.ehr_data.tokenizer_pretrain.dx_voc, self.ehr_data.tokenizer_pretrain.rx_voc)
        self.cls_pretrain = SelfSupervisedHead(config, self.dx_voc_size, self.rx_voc_size)

        # predict part
        self.dense = nn.ModuleList([MappingHead(config), MappingHead(config)])
        self.cls_predict = nn.Sequential(
            nn.Linear(3 * config.hidden_size, 2 * config.hidden_size), nn.ReLU(),
            nn.Linear(2 * config.hidden_size, len(self.ehr_data.tokenizer_predict.rx_voc_multi.word2idx)))

        # todo verify this part
        # embedding for BERT, sum of positional, segment, token embeddings
        if config.graph:
            assert self.ehr_data.tokenizer_pretrain is not None
            assert self.ehr_data.tokenizer_pretrain.dx_voc is not None
            assert self.ehr_data.tokenizer_pretrain.rx_voc is not None
            self.embedding = FuseEmbeddings(config, self.ehr_data.tokenizer_pretrain.dx_voc,
                                            self.ehr_data.tokenizer_pretrain.rx_voc)
        else:
            self.embedding = BertEmbeddings(config)

        self.learning_rate = args.learning_rate
        self.metric_report = metric_report(args.threshold)

        # multi-layers transformer blocks, deep network
        # todo refer to self.bert not to a new copy
        # self.transformer_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.mode = None
        self.max_metric = 0
        self.apply(self.bert.init_bert_weights)

    @staticmethod
    def add_model_specific_args(parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument("--use_pretrain",
                            default=False,
                            action='store_true',
                            help="if use ontology embedding")
        parser.add_argument("--learning_rate",
                            default=5e-4,
                            type=float,
                            help="The initial learning rate for Adam.")
        parser.add_argument("--graph",
                            default=False,
                            action='store_true',
                            help="if use ontology embedding")
        parser.add_argument("--threshold",
                            default=0.3,
                            type=float,
                            help="threshold for metrics eval.")
        return parser

    def set_mode(self, mode: BertMode):
        self.mode = mode
        self.ehr_data.mode = mode

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @staticmethod
    def compute_pretrain_loss(dx2dx, rx2dx, dx2rx, rx2rx, dx_labels, rx_labels):
        return F.binary_cross_entropy_with_logits(dx2dx, dx_labels) + \
               F.binary_cross_entropy_with_logits(rx2dx, dx_labels) + \
               F.binary_cross_entropy_with_logits(dx2rx, rx_labels) + \
               F.binary_cross_entropy_with_logits(rx2rx, rx_labels)

    def pretrain_fw(self, inputs) -> Tuple:
        _, dx_bert_pool = self.bert(inputs[:, 0, :], torch.zeros((inputs.size(0), inputs.size(2))).long().to(inputs.device))
        _, rx_bert_pool = self.bert(inputs[:, 1, :], torch.zeros((inputs.size(0), inputs.size(2))).long().to(inputs.device))
        return self.cls_pretrain(dx_bert_pool, rx_bert_pool)

    def predict_fw(self, input_ids, n_rx_labels:int):
        token_types_ids = torch.cat([torch.zeros((1, input_ids.size(1))), torch.ones(
            (1, input_ids.size(1)))], dim=0).long().to(input_ids.device)
        token_types_ids = token_types_ids.repeat(
            1 if input_ids.size(0) // 2 == 0 else input_ids.size(0) // 2, 1)
        # bert_pool: (2*adm, H)
        _, bert_pool = self.bert(input_ids, token_types_ids)
        bert_pool = bert_pool.view(2, -1, bert_pool.size(1))  # (2, adm, H)
        dx_bert_pool = self.dense[0](bert_pool[0])  # (adm, H)
        rx_bert_pool = self.dense[1](bert_pool[1])  # (adm, H)

        # mean and concat for rx prediction task
        rx_logits = []
        for i in range(n_rx_labels):
            # mean
            dx_mean = torch.mean(dx_bert_pool[0:i + 1, :], dim=0, keepdim=True)
            rx_mean = torch.mean(rx_bert_pool[0:i + 1, :], dim=0, keepdim=True)
            # concat
            concat = torch.cat(
                [dx_mean, rx_mean, dx_bert_pool[i + 1, :].unsqueeze(dim=0)], dim=-1)
            rx_logits.append(self.cls_predict(concat))

        rx_logits = torch.cat(rx_logits, dim=0)
        return rx_logits

    def forward(self, input_ids, n_rx_labels=None):
        if self.mode == BertMode.Pretrain:
            return self.pretrain_fw(input_ids)
        elif self.mode == BertMode.Predict:
            return self.predict_fw(input_ids, n_rx_labels)
        else:
            raise NotImplementedError()

    def training_step(self, batch):
        input_ids, dx_labels, rx_labels = batch
        if self.mode == BertMode.Pretrain:
            dx2dx, rx2dx, dx2rx, rx2rx = self(input_ids)
            loss = LitGBert.compute_pretrain_loss(dx2dx, rx2dx, dx2rx, rx2rx, dx_labels, rx_labels)
            self.log(f"train_loss_pretrain", loss)
            return loss
        elif self.mode == BertMode.Predict:
            input_ids, rx_labels = input_ids.squeeze(dim=0), rx_labels.squeeze(dim=0)
            n_rx_labels = rx_labels.size(0)
            rx_logits = self(input_ids, n_rx_labels)
            loss = F.binary_cross_entropy_with_logits(rx_logits, rx_labels)
            self.log(f"train_loss_predict", loss)
            return loss
        else:
            raise NotImplementedError()

    def validation_step(self, batch, batch_idx):
        input_ids, dx_labels, rx_labels = batch
        if self.mode == BertMode.Pretrain:
            dx2dx, rx2dx, dx2rx, rx2rx = self(input_ids)
            acc_container = {
                'dx2dx': self.metric_report(t2n(dx2dx), t2n(dx_labels)),
                'rx2dx': self.metric_report(t2n(rx2dx), t2n(dx_labels)),
                'dx2rx': self.metric_report(t2n(dx2rx), t2n(rx_labels)),
                'rx2rx': self.metric_report(t2n(rx2rx), t2n(rx_labels))
            }
            for cat_key in acc_container:
                for metric_key in acc_container[cat_key]:
                    self.log(f"val_{cat_key}_{metric_key}", acc_container[cat_key][metric_key])
            self.log(f"val_loss", LitGBert.compute_pretrain_loss(dx2dx, rx2dx, dx2rx, rx2rx, dx_labels, rx_labels))
        elif self.mode == BertMode.Predict:
            input_ids, rx_labels = input_ids.squeeze(dim=0), rx_labels.squeeze(dim=0)
            n_rx_labels = rx_labels.size(0)
            rx_probs = torch.sigmoid(self(input_ids, n_rx_labels))
            metrics = self.metric_report(t2n(rx_probs), t2n(rx_labels))
            for metric_key in metrics:
                self.log(f"val_predict_{metric_key}", metrics[metric_key])
        else:
            raise NotImplementedError()

    def test_step(self, batch, batch_idx):
        input_ids, dx_labels, rx_labels = batch
        if self.mode == BertMode.Pretrain:
            dx2dx, rx2dx, dx2rx, rx2rx = self(input_ids)
            acc_container = {
                'dx2dx': self.metric_report(t2n(dx2dx), t2n(dx_labels)),
                'rx2dx': self.metric_report(t2n(rx2dx), t2n(dx_labels)),
                'dx2rx': self.metric_report(t2n(dx2rx), t2n(rx_labels)),
                'rx2rx': self.metric_report(t2n(rx2rx), t2n(rx_labels))
            }
            for cat_key in acc_container:
                for metric_key in acc_container[cat_key]:
                    self.log(f"test_{cat_key}_{metric_key}", acc_container[cat_key][metric_key])
        elif self.mode == BertMode.Predict:
            input_ids, rx_labels = input_ids.squeeze(dim=0), rx_labels.squeeze(dim=0)
            n_rx_labels = rx_labels.size(0)
            rx_probs = torch.sigmoid(self(input_ids, n_rx_labels))
            metrics = self.metric_report(t2n(rx_probs), t2n(rx_labels))
            for metric_key in metrics:
                self.log(f"test_predict_{metric_key}", metrics[metric_key])
        else:
            raise NotImplementedError()

