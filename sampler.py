import random
from torch.utils.data import Sampler

class PKSampler(Sampler):
    def __init__(self, labels, p=8, k=4):
        super().__init__()
        self.labels = labels
        self.p = p
        self.k = k

        self.index_dict = {}
        for idx, label in enumerate(labels):
            self.index_dict.setdefault(label, []).append(idx)

        self.labels_set = list(self.index_dict.keys())

    def __iter__(self):
        batch = []
        labels = random.sample(self.labels_set, self.p)

        for label in labels:
            idxs = random.sample(self.index_dict[label], self.k)
            batch.extend(idxs)

        return iter(batch)

    def __len__(self):
        return self.p * self.k
