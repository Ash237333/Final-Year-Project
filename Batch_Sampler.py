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

    def __iter__(self):
        curr_batch = []
        curr_batch_seq_length = 0

        for idx in self.sorted_indexes:
            length = self.lengths[idx]
            if length > curr_batch_seq_length:
                curr_batch_seq_length = length
            if curr_batch_seq_length * (len(curr_batch) + 1) <= self.target_total_tokens:
                curr_batch.append(int(idx))
            else:
                yield curr_batch
                curr_batch = [int(idx)]
                curr_batch_seq_length = length
        if curr_batch:
            yield curr_batch


    def __len__(self):
        num_batches = 0
        curr_batch = []
        curr_batch_seq_length = 0

        for idx in self.sorted_indexes:
            length = self.lengths[idx]
            if length > curr_batch_seq_length:
                curr_batch_seq_length = length
            if curr_batch_seq_length * (len(curr_batch) + 1) <= self.target_total_tokens:
                curr_batch.append(int(idx))
            else:
                num_batches += 1
                curr_batch = [int(idx)]
                curr_batch_seq_length = length
        if curr_batch:
            num_batches += 1
        return num_batches
