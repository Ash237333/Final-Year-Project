from torch.utils.data import Sampler
from typing import Iterable
import numpy as np

class Custom_Sampler(Sampler):
    def __init__(self, data_source: Iterable, target_total_tokens):
        super().__init__(data_source)
        self.lengths = []
        self.target_total_tokens = target_total_tokens
        self.batches = []

        for i in data_source:
            label_length = len(i["labels"])
            input_length = len(i["input_ids"])
            length = max(label_length, input_length)
            self.lengths.append(length)

        self.sorted_indexes = np.argsort(self.lengths)[::-1]

        self._get_batches()

    def __iter__(self):
        # Randomly shuffles order of the batches each epoch
        np.random.shuffle(self.batches)
        for batch in self.batches:
            yield batch


    def _get_batches(self):
        curr_batch = []
        curr_batch_seq_length = 0

        for idx in self.sorted_indexes:
            length = self.lengths[idx]
            if length > curr_batch_seq_length:
                curr_batch_seq_length = length
            if curr_batch_seq_length * (len(curr_batch) + 1) <= self.target_total_tokens:
                curr_batch.append(int(idx))
            else:
                self.batches.append(curr_batch)
                curr_batch = [int(idx)]
                curr_batch_seq_length = length
        if curr_batch:
            self.batches.append(curr_batch)


    def __len__(self):
        return len(self.batches)
