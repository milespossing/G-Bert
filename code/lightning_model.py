from typing import Dict, Tuple

import pytorch_lightning as pl
import numpy as np
import torch
import enum
from torch import nn
from torch.nn import functional as F

from lightning_data_module import Tokenizer
from config import BertConfig
from bert_models import FuseEmbeddings, BertEmbeddings, TransformerBlock, LayerNorm, BERT
from predictive_models import SelfSupervisedHead, MappingHead
from utils import t2n, multi_label_metric


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


class LitGBert(pl.LightningModule, ):
    def __init__(self, config: BertConfig, tokenizer_pretrain: Tokenizer, tokenizer_predict: Tokenizer,
                 learning_rate: float, eval_threshold: float):
        super().__init__()
        assert tokenizer_predict.dx_voc.word2idx == tokenizer_pretrain.dx_voc.word2idx
        assert tokenizer_predict.rx_voc.word2idx == tokenizer_pretrain.rx_voc.word2idx
        self.dx_voc_size = len(tokenizer_pretrain.dx_voc.word2idx)
        self.rx_voc_size = len(tokenizer_pretrain.rx_voc.word2idx)

        # pretrain part
        self.bert = BERT(config, tokenizer_pretrain.dx_voc, tokenizer_pretrain.rx_voc)
        self.cls_pretrain = SelfSupervisedHead(config, self.dx_voc_size, self.rx_voc_size)

        # predict part
        self.dense = nn.ModuleList([MappingHead(config), MappingHead(config)])
        self.cls_predict = nn.Sequential(nn.Linear(3 * config.hidden_size, 2 * config.hidden_size),
                                         nn.ReLU(),
                                         nn.Linear(2*config.hidden_size, len(tokenizer_predict.rx_voc_multi.word2idx)))

        # todo verify this part
        # embedding for BERT, sum of positional, segment, token embeddings
        if config.graph:
            assert tokenizer_pretrain is not None
            assert tokenizer_pretrain.dx_voc is not None
            assert tokenizer_pretrain.rx_voc is not None
            self.embedding = FuseEmbeddings(config, tokenizer_pretrain.dx_voc, tokenizer_pretrain.rx_voc)
        else:
            self.embedding = BertEmbeddings(config)

        self.learning_rate = learning_rate
        self.config = config
        self.metric_report = metric_report(eval_threshold)

        # multi-layers transformer blocks, deep network
        # todo refer to self.bert not to a new copy
        # self.transformer_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])
        self.mode = None
        self.apply(self.bert.init_bert_weights)

    def set_mode(self, mode: BertMode):
        self.mode = mode

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

        # input_ids, dx_labels, rx_labels = batch
        # input_ids, dx_labels, rx_labels = input_ids.squeeze(dim=0), dx_labels.squeeze(dim=0), rx_labels.squeeze(dim=0)
        # rx_logits = self.predict_forward(input_ids, rx_labels)
        # loss = F.binary_cross_entropy_with_logits(rx_logits, rx_labels)

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
            return LitGBert.compute_pretrain_loss(dx2dx, rx2dx, dx2rx, rx2rx, dx_labels, rx_labels)
        elif self.mode == BertMode.Predict:
            input_ids, rx_labels = input_ids.squeeze(dim=0), rx_labels.squeeze(dim=0)
            n_rx_labels = rx_labels.size(0)
            rx_logits = self(input_ids, n_rx_labels)
            return F.binary_cross_entropy_with_logits(rx_logits, rx_labels)
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

