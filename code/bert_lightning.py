import pytorch_lightning as pl
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from config import BertConfig
from bert_models import FuseEmbeddings, BertEmbeddings, TransformerBlock, LayerNorm
from predictive_models import SelfSupervisedHead, MappingHead
from utils import t2n, multi_label_metric


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
        self.learning_rate = learning_rate
        if config.graph:
            assert tokenizer is not None
            assert tokenizer.dx_voc is not None
            assert tokenizer.rx_voc is not None

        self.config = config

        # TODO: add this as a config value
        self.metric_report = metric_report(0.5)

        # embedding for BERT, sum of positional, segment, token embeddings
        if config.graph:
            self.embedding = FuseEmbeddings(config, tokenizer.dx_voc, tokenizer.rx_voc)
        else:
            self.embedding = BertEmbeddings(config)

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_hidden_layers)])

        self.apply(self.init_bert_weights)

    def forward(self, x, token_type_ids):
        # attention masking for padded token
        # torch.ByteTensor([batch_size, 1, seq_len, seq_len)
        mask = (x > 1).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, token_type_ids)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x, x[:, 0]

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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def validation_epoch_end(self, outputs) -> None:
        dx2dx_y_preds, rx2dx_y_preds, dx2rx_y_preds, rx2rx_y_preds, dx_y_trues, rx_y_trues = map(list,
                                                                                                 zip(*outputs))

        dx2dx = self.metric_report(
            np.concatenate(dx2dx_y_preds, axis=0), np.concatenate(dx_y_trues, axis=0))
        print('')
        print('dx2dx')
        print(dx2dx)
        rx2dx = self.metric_report(
            np.concatenate(rx2dx_y_preds, axis=0), np.concatenate(dx_y_trues, axis=0))
        print('rx2dx')
        print(rx2dx)
        dx2rx = self.metric_report(
            np.concatenate(dx2rx_y_preds, axis=0), np.concatenate(rx_y_trues, axis=0))
        print('dx2rx')
        print(dx2rx)
        rx2rx = self.metric_report(
            np.concatenate(rx2rx_y_preds, axis=0), np.concatenate(rx_y_trues, axis=0))
        print('rx2rx')
        print(rx2dx)


class LitPretrain(LitBert):
    def __init__(self, config: BertConfig, tokenizer, learning_rate, ):
        super().__init__(config, tokenizer, learning_rate)
        self.cls = SelfSupervisedHead(config, len(tokenizer.dx_voc.word2idx), len(tokenizer.rx_voc.word2idx))

    def training_step(self, batch):
        inputs, dx_labels, rx_labels = batch
        # TODO: Might need to sqeeze inputs here like in ./run_gbert.py:418
        inputs, dx_labels, rx_labels = inputs.squeeze(), dx_labels.squeeze(), rx_labels.squeeze(dim=0)
        # inputs (B, 2, max_len)
        # bert_pool (B, hidden)
        _, dx_bert_pool = self.forward(inputs[:, 0, :], torch.zeros(
            (inputs.size(0), inputs.size(2))).long().to(inputs.device))
        _, rx_bert_pool = self.forward(inputs[:, 1, :], torch.zeros(
            (inputs.size(0), inputs.size(2))).long().to(inputs.device))

        dx2dx, rx2dx, dx2rx, rx2rx = self.cls(dx_bert_pool, rx_bert_pool)
        # compute loss
        return compute_loss(dx2dx, rx2dx, dx2rx, rx2rx, dx_labels, rx_labels)

    def validation_step(self, batch, batch_idx):
        """
        TODO: This returns a very small number right now. I should find a way to get the output of _all_ batches and
              compute the metrics from that point.
        """
        inputs, dx_labels, rx_labels = batch
        inputs, dx_labels, rx_labels = inputs.squeeze(), dx_labels.squeeze(), rx_labels.squeeze(dim=0)
        # inputs (B, 2, max_len)
        # bert_pool (B, hidden)
        _, dx_bert_pool = self.forward(inputs[:, 0, :], torch.zeros(
            (inputs.size(0), inputs.size(2))).long().to(inputs.device))
        _, rx_bert_pool = self.forward(inputs[:, 1, :], torch.zeros(
            (inputs.size(0), inputs.size(2))).long().to(inputs.device))

        dx2dx, rx2dx, dx2rx, rx2rx = self.cls(dx_bert_pool, rx_bert_pool)
        dx2dx, rx2dx, dx2rx, rx2rx = torch.sigmoid(dx2dx), torch.sigmoid(rx2dx), torch.sigmoid(dx2rx), torch.sigmoid(
            rx2rx)
        return t2n(dx2dx), t2n(rx2dx), t2n(dx2rx), t2n(rx2rx), t2n(dx_labels), t2n(rx_labels)


class LitPredict(LitBert):
    def __init__(self):
        pass
