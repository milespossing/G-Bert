import os
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
    def __init__(self, is_pretrain: bool, data_dir: str = "./data", max_seq_len=55, batch_size=10):
        super().__init__()
        self.train_dataset, self.eval_dataset, self.test_dataset = (None, None, None)
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        self.is_pretrain = is_pretrain
        self.batch_size = batch_size if is_pretrain else 1
        # load tokenizer
        self.tokenizer = Tokenizer(self.data_dir, self.is_pretrain)

    def prepare_data(self):
        # TODO: here we download & save data (part of load_dataset from orig code)
        pass

    def setup(self, stage: Optional[str] = None):
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

        if self.is_pretrain:
            data_multi = pd.read_pickle(os.path.join(self.data_dir, 'data-multi-visit.pkl')).iloc[:, :4]
            data_single = pd.read_pickle(os.path.join(self.data_dir, 'data-single-visit.pkl'))
            self.train_dataset = DatasetPretrain(pd.concat([data_single, load_ids(data_multi, ids_file[0])]), self.tokenizer, self.max_seq_len)
            self.eval_dataset = DatasetPretrain(load_ids(data_multi, ids_file[1]), self.tokenizer, self.max_seq_len)
            self.test_dataset = DatasetPretrain(load_ids(data_multi, ids_file[2]), self.tokenizer, self.max_seq_len)
        else:  # Prediction task
            data_multi = pd.read_pickle(os.path.join(self.data_dir, 'data-multi-visit.pkl'))
            self.train_dataset, self.eval_dataset, self.test_dataset = \
                tuple(map(lambda x: DatasetPrediction(load_ids(data_multi, x), self.tokenizer, self.max_seq_len), ids_file))
        pass

    def train_dataloader(self, num_workers=6):
        return DataLoader(self.train_dataset,
                          sampler=RandomSampler(self.train_dataset),
                          batch_size=self.batch_size, num_workers=num_workers)

    def val_dataloader(self, num_workers=1):
        return DataLoader(self.eval_dataset,
                          sampler=SequentialSampler(self.eval_dataset),
                          batch_size=self.batch_size,
                          num_workers=num_workers)

    def test_dataloader(self, num_workers=1):
        return DataLoader(self.test_dataset,
                          sampler=SequentialSampler(self.test_dataset),
                          batch_size=self.batch_size,
                          num_workers=num_workers)


if __name__ == '__main__':
    for pretrain in [False, True]:
        ehr_data = EHRDataModule(is_pretrain=pretrain, data_dir='../data', batch_size=64)
        ehr_data.setup()
        _train = iter(ehr_data.train_dataloader())
        next(_train)
        _val = iter(ehr_data.val_dataloader())
        next(_val)
        _test = iter(ehr_data.test_dataloader())
        next(_test)
        del ehr_data, _train, _val, _test
    print('all is well')
