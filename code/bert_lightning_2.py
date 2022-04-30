import os
import pytorch_lightning as pl
import numpy as np
import torch
import enum
from torch import nn
from torch.nn import functional as F
from config import BertConfig
from bert_models import FuseEmbeddings, BertEmbeddings, TransformerBlock, LayerNorm
from predictive_models import SelfSupervisedHead, MappingHead
from utils import t2n, multi_label_metric

CONFIG_NAME = 'bert_config.json'

class MetricData:
    def __init__(self, jaccard, f1, prauc):
        self.jaccard = jaccard
        self.f1 = f1
        self.prauc = prauc

    def __str__(self):
        return f'jaccard: {self.jaccard}; f1: {self.f1}; prauc: {self.prauc}'


def metric_report(threshold=0.5):
    def get_metrics(y_pred, y_true):
        y_prob = y_pred.copy()
        y_pred[y_pred > threshold] = 1
        y_pred[y_pred <= threshold] = 0
        ja, prauc, avg_p, avg_r, avg_f1 = multi_label_metric(
            y_true, y_pred, y_prob)
        return MetricData(ja, avg_f1, prauc)

    return lambda pred, true: get_metrics(pred, true)


def compute_loss(dx2dx, rx2dx, dx2rx, rx2rx, dx_labels, rx_labels):
    return F.binary_cross_entropy_with_logits(dx2dx, dx_labels) + \
           F.binary_cross_entropy_with_logits(rx2dx, dx_labels) + \
           F.binary_cross_entropy_with_logits(dx2rx, rx_labels) + \
           F.binary_cross_entropy_with_logits(rx2rx, rx_labels)


class LitBert(pl.LightningModule):
    def __init__(self, config: BertConfig, tokenizer, learning_rate):
        super().__init__()
        self.config = config
        self.learning_rate = learning_rate
        self.dense = nn.ModuleList([MappingHead(config), MappingHead(config)])
        # TODO: Make this configurable    vvvvv
        self.metric_report = metric_report(0.5)
        # embedding for BERT, sum of positional, segment, token embeddings
        if config.graph:
            self.embedding = FuseEmbeddings(config, tokenizer.dx_voc, tokenizer.rx_voc)
        else:
            self.embedding = BertEmbeddings(config)

        self.transformer_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(
                mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def bert(self, x, token_type_ids):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        mask = (x > 1).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, token_type_ids)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x, x[:, 0]

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class LitBertPretrain(LitBert):
    def __init__(self, config: BertConfig, tokenizer, learning_rate):
        super().__init__(config, tokenizer, learning_rate)
        self.cls = SelfSupervisedHead(config, len(tokenizer.dx_voc.word2idx), len(tokenizer.rx_voc.word2idx))
        if config.graph:
            assert tokenizer is not None
            assert tokenizer.dx_voc is not None
            assert tokenizer.rx_voc is not None

        self.apply(self.init_bert_weights)

    def forward(self, inputs):
        # inputs (B, 2, max_len)
        # bert_pool (B, hidden)
        _, dx_bert_pool = self.bert(inputs[:, 0, :], torch.zeros(
            (inputs.size(0), inputs.size(2))).long().to(inputs.device))
        _, rx_bert_pool = self.bert(inputs[:, 1, :], torch.zeros(
            (inputs.size(0), inputs.size(2))).long().to(inputs.device))
        return self.cls(dx_bert_pool, rx_bert_pool)

    def training_step(self, batch):
        inputs, dx_labels, rx_labels = batch
        inputs, dx_labels, rx_labels = inputs.squeeze(), dx_labels.squeeze(), rx_labels.squeeze(dim=0)
        dx2dx, rx2dx, dx2rx, rx2rx = self.forward(inputs)

        # compute loss
        return compute_loss(dx2dx, rx2dx, dx2rx, rx2rx, dx_labels, rx_labels)

    def validation_step(self, batch, batch_idx):
        inputs, dx_labels, rx_labels = batch
        inputs, dx_labels, rx_labels = inputs.squeeze(), dx_labels.squeeze(), rx_labels.squeeze(dim=0)
        # inputs (B, 2, max_len)
        # bert_pool (B, hidden)
        dx2dx, rx2dx, dx2rx, rx2rx = self.forward(inputs)
        dx2dx, rx2dx, dx2rx, rx2rx = torch.sigmoid(dx2dx), torch.sigmoid(rx2dx), torch.sigmoid(dx2rx), torch.sigmoid(
            rx2rx)
        return dx2dx, rx2dx, dx2rx, rx2rx, dx_labels, rx_labels

    def validation_epoch_end(self, outputs):
        dx2dx_y_preds, rx2dx_y_preds, dx2rx_y_preds, rx2rx_y_preds, dx_y_trues, rx_y_trues = map(list,
                                                                                                 zip(*outputs))
        dx_y_trues = t2n(torch.cat(dx_y_trues, dim=0))
        rx_y_trues = t2n(torch.cat(rx_y_trues, dim=0))

        dx2dx = self.metric_report(
            t2n(torch.cat(dx2dx_y_preds, dim=0)), dx_y_trues)
        print('')
        print('dx2dx')
        print(dx2dx)
        rx2dx = self.metric_report(
            t2n(torch.cat(rx2dx_y_preds, dim=0)), dx_y_trues)
        print('rx2dx')
        print(rx2dx)
        dx2rx = self.metric_report(
            t2n(torch.cat(dx2rx_y_preds, dim=0)), rx_y_trues)
        print('dx2rx')
        print(dx2rx)
        rx2rx = self.metric_report(
            t2n(torch.cat(rx2rx_y_preds, dim=0)), rx_y_trues)
        print('rx2rx')
        print(rx2rx)


class LitBertPredict(LitBert):
    def __init__(self, config: BertConfig, tokenizer, learning_rate):
        super().__init__(config, tokenizer, learning_rate)
        self.cls = nn.Sequential(nn.Linear(3 * config.hidden_size, 2 * config.hidden_size),
                                         nn.ReLU(),
                                         nn.Linear(2*config.hidden_size, len(tokenizer.rx_voc_multi.word2idx)))

        if config.graph:
            assert tokenizer is not None
            assert tokenizer.dx_voc is not None
            assert tokenizer.rx_voc is not None

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, rx_length):
        token_types_ids = torch.cat([torch.zeros((1, input_ids.size(1))), torch.ones(
            (1, input_ids.size(1)))], dim=0).long().to(input_ids.device)
        token_types_ids = token_types_ids.repeat(
            1 if input_ids.size(0)//2 == 0 else input_ids.size(0)//2, 1)
        _, bert_pool = self.bert(input_ids, token_types_ids)
        loss = 0
        bert_pool = bert_pool.view(2, -1, bert_pool.size(1))  # (2, adm, H)
        dx_bert_pool = self.dense[0](bert_pool[0])  # (adm, H)
        rx_bert_pool = self.dense[1](bert_pool[1])  # (adm, H)
        rx_logits = []
        for i in range(rx_length):
            # mean
            dx_mean = torch.mean(dx_bert_pool[0:i+1, :], dim=0, keepdim=True)
            rx_mean = torch.mean(rx_bert_pool[0:i+1, :], dim=0, keepdim=True)
            # concat
            concat = torch.cat(
                [dx_mean, rx_mean, dx_bert_pool[i+1, :].unsqueeze(dim=0)], dim=-1)
            rx_logits.append(self.cls(concat))

        return torch.cat(rx_logits, dim=0)

    def training_step(self, batch):
        input_ids, dx_labels, rx_labels = batch
        input_ids, dx_labels, rx_labels = input_ids.squeeze(
            dim=0), dx_labels.squeeze(dim=0), rx_labels.squeeze(dim=0)
        rx_logits = self.forward(input_ids, rx_labels.size(0))

    def validation_step(self, batch, batch_idx):
        input_ids, dx_labels, rx_labels = batch
        input_ids, dx_labels, rx_labels = input_ids.squeeze(
            dim=0), dx_labels.squeeze(dim=0), rx_labels.squeeze(dim=0)
        rx_logits = self.forward(input_ids, rx_labels.size(0))
        return torch.sigmoid(rx_logits), rx_labels

    def validation_epoch_end(self, outputs):
        y_pred, y_true = map(list, zip(*outputs))
        y_pred, y_true = torch.cat(y_pred), torch.cat(y_true)
        metrics = self.metric_report(t2n(y_pred), t2n(y_true))
        print('')
        print(metrics)

