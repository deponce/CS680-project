import torch
import json
import os

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