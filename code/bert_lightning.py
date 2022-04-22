import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from config import BertConfig
from bert_models import FuseEmbeddings, BertEmbeddings, TransformerBlock, LayerNorm
from predictive_models import SelfSupervisedHead

class LitBert(pl.LightningModule):
    def __init__(self, config: BertConfig, dx_voc=None, rx_voc=None, learning_rate=5e-4):
        super().__init__()
        self.learning_rate = learning_rate
        if config.graph:
            assert dx_voc is not None
            assert rx_voc is not None

        self.config = config
        self.cls = SelfSupervisedHead(config, len(dx_voc.word2idx), len(rx_voc.word2idx))

        # embedding for BERT, sum of positional, segment, token embeddings
        if config.graph:
            self.embedding = FuseEmbeddings(config, dx_voc, rx_voc)
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

    def forward(self, x, token_type_ids=None, input_positions=None, input_sides=None):
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
        # inputs (B, 2, max_len)
        # bert_pool (B, hidden)
        _, dx_bert_pool = self.forward(inputs[:, 0, :], torch.zeros(
            (inputs.size(0), inputs.size(2))).long().to(inputs.device))
        _, rx_bert_pool = self.forward(inputs[:, 1, :], torch.zeros(
            (inputs.size(0), inputs.size(2))).long().to(inputs.device))

        dx2dx, rx2dx, dx2rx, rx2rx = self.cls(dx_bert_pool, rx_bert_pool)
        # compute loss
        return F.binary_cross_entropy_with_logits(dx2dx, dx_labels) + \
               F.binary_cross_entropy_with_logits(rx2dx, dx_labels) + \
               F.binary_cross_entropy_with_logits(dx2rx, rx_labels) + \
               F.binary_cross_entropy_with_logits(rx2rx, rx_labels)
