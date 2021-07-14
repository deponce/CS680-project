import torch
import json
import os
import numpy as np

class ZhEnDataset(torch.utils.data.IterableDataset):
    def __init__(self, split, start, end):
        super(ZhEnDataset).__init__()


        self.start = start
        self.end = end

        if split == 'train':
            _file = os.path.join(".", "translation2019zh", "translation2019zh_train.json")
        elif split == 'valid':
            _file = os.path.join(".", "translation2019zh", "translation2019zh_valid.json")

        self.pairs = []
        for i, line in enumerate( open(_file, mode='r', encoding='utf-8')):
            if i < start:
                continue
            d = json.loads(line)
            self.pairs.append( (d['english'] + '\n', d['chinese'] + '\n') )
            if i == end-1 :
                break

    def __iter__(self):
        return iter(self.pairs)

    def __len__(self):
        return self.end - self.start


class BaggingDataset(torch.utils.data.IterableDataset):
    def __init__(self, size):
        super(BaggingDataset).__init__()

        self.size = size

        # for training only
        _file = os.path.join(".", "translation2019zh", "translation2019zh_train.json")

        # load the whole training set
        self.pairs = []
        for i, line in enumerate( open(_file, mode='r', encoding='utf-8')):
            d = json.loads(line)
            self.pairs.append( (d['english'] + '\n', d['chinese'] + '\n') )

        self.pairs_sampled = []

    def resample(self):
        self.pairs_sampled = [self.pairs[i] for i in np.random.choice(len(self.pairs), size=self.size, replace=True, p=None)]

    def __iter__(self):
        return iter(self.pairs_sampled)

    def __len__(self):
        return self.size