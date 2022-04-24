import pytorch_lightning as pl
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

def metric_report(threshold=0.5):
    def get_metrics(y_pred, y_true):
        y_prob = t2n(y_pred).copy()
        y_pred = t2n(y_pred)
        y_true = t2n(y_true).copy()
        y_pred[y_pred > threshold] = 1
        y_pred[y_pred <= threshold] = 0
        ja, prauc, avg_p, avg_r, avg_f1 = multi_label_metric(
            y_true, y_pred, y_prob)
        return MetricData(ja, avg_f1, prauc)
    return lambda pred, true: get_metrics(pred, true)
    pass

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
        self.dense = nn.ModuleList([MappingHead(config), MappingHead(config)])
        self.cls_train = SelfSupervisedHead(config, len(tokenizer.dx_voc.word2idx), len(tokenizer.rx_voc.word2idx))
        self.cls_pred = nn.Sequential(nn.Linear(3 * config.hidden_size, 2 * config.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(2 * config.hidden_size, len(tokenizer.rx_voc_multi.word2idx)))

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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

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

        dx2dx, rx2dx, dx2rx, rx2rx = self.cls_train(dx_bert_pool, rx_bert_pool)
        # compute loss
        return compute_loss(dx2dx, rx2dx, dx2rx, rx2rx, dx_labels, rx_labels)

    def predict_forward(self, input_ids):
        token_types_ids = torch.cat([torch.zeros((1, input_ids.size(1))), torch.ones(
            (1, input_ids.size(1)))], dim=0).long().to(input_ids.device)
        token_types_ids = token_types_ids.repeat(
            1 if input_ids.size(0) // 2 == 0 else input_ids.size(0) // 2, 1)
        # bert_pool: (2*adm, H)
        _, bert_pool = self.forward(input_ids, token_types_ids)
        loss = 0
        bert_pool = bert_pool.view(2, -1, bert_pool.size(1))  # (2, adm, H)
        dx_bert_pool = self.dense[0](bert_pool[0])  # (adm, H)
        rx_bert_pool = self.dense[1](bert_pool[1])  # (adm, H)
        return dx_bert_pool, rx_bert_pool

    def validation_step(self, batch, batch_idx):
        inputs, dx_labels, rx_labels = batch
        # TODO: Might need to sqeeze inputs here like in ./run_gbert.py:418
        inputs, dx_labels, rx_labels = inputs.squeeze(), dx_labels.squeeze(), rx_labels.squeeze(dim=0)
        # inputs (B, 2, max_len)
        # bert_pool (B, hidden)
        _, dx_bert_pool = self.forward(inputs[:, 0, :], torch.zeros(
            (inputs.size(0), inputs.size(2))).long().to(inputs.device))
        _, rx_bert_pool = self.forward(inputs[:, 1, :], torch.zeros(
            (inputs.size(0), inputs.size(2))).long().to(inputs.device))

        dx2dx, rx2dx, dx2rx, rx2rx = self.cls_train(dx_bert_pool, rx_bert_pool)
        dx2dx, rx2dx, dx2rx, rx2rx = torch.sigmoid(dx2dx), torch.sigmoid(rx2dx), torch.sigmoid(dx2rx), torch.sigmoid(rx2rx)

        dx2dx_metrics = self.metric_report(dx2dx, dx_labels)
        rx2dx_metrics = self.metric_report(rx2dx, dx_labels)
        dx2rx_metrics = self.metric_report(dx2rx, rx_labels)
        rx2rx_metrics = self.metric_report(rx2rx, rx_labels)

        self.log_dict({
            'dx2dx_jaccard': dx2dx_metrics.jaccard,
            'dx2dx_f1': dx2dx_metrics.f1,
            'dx2dx_prauc': dx2dx_metrics.prauc,
            'rx2dx_jaccard': rx2dx_metrics.jaccard,
            'rx2dx_f1': rx2dx_metrics.f1,
            'rx2dx_prauc': rx2dx_metrics.prauc,
            'dx2rx_jaccard': dx2rx_metrics.jaccard,
            'dx2rx_f1': dx2rx_metrics.f1,
            'dx2rx_prauc': dx2rx_metrics.prauc,
            'rx2rx_jaccard': rx2rx_metrics.jaccard,
            'rx2rx_f1': rx2rx_metrics.f1,
            'rx2rx_prauc': rx2rx_metrics.prauc,
        })

    def predict_step(self, batch, **kwargs):
        input_ids, _, rx_labels = batch
        token_types_ids = torch.cat([torch.zeros((1, input_ids.size(1))), torch.ones(
            (1, input_ids.size(1)))], dim=0).long().to(input_ids.device)
        token_types_ids = token_types_ids.repeat(
            1 if input_ids.size(0) // 2 == 0 else input_ids.size(0) // 2, 1)
        _, bert_pool = self.forward(input_ids, token_types_ids)
        bert_pool = bert_pool.view(2, -1, bert_pool.size(1))  # (2, adm, H)
        dx_bert_pool = self.dense[0](bert_pool[0])  # (adm, H)
        rx_bert_pool = self.dense[1](bert_pool[1])  # (adm, H)

        # mean and concat for rx prediction task
        rx_logits = []
        for i in range(rx_labels.size(0)):
            # mean
            dx_mean = torch.mean(dx_bert_pool[0:i + 1, :], dim=0, keepdim=True)
            rx_mean = torch.mean(rx_bert_pool[0:i + 1, :], dim=0, keepdim=True)
            # concat
            concat = torch.cat(
                [dx_mean, rx_mean, dx_bert_pool[i + 1, :].unsqueeze(dim=0)], dim=-1)
            rx_logits.append(self.cls(concat))

        return torch.cat(rx_logits, dim=0)

    def test_step(self, batch, batch_idx):
        _, _, rx_labels = batch
        rx_logits = self.predict_step(batch)
        loss = F.binary_cross_entropy_with_logits(rx_logits, rx_labels)
        self.log_dict({'loss': loss})

    # def validation_step(self, batch):
    #     inputs, dx_labels, rx_labels = batch
    #
    #     with torch.no_grad():
    #         rx_logits = self.predict_step(batch)
    #         rx_y_preds.append(t2n(torch.sigmoid(rx_logits)))
    #         rx_y_trues.append(t2n(rx_labels))
