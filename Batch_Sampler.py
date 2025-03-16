from torch.utils.data import Sampler
from typing import Iterable
import numpy as np

class Custom_Sampler(Sampler):
    def __init__(self, data_source: Iterable, target_total_tokens):
        super().__init__(data_source)
        self.lengths = []
        self.target_total_tokens = target_total_tokens
        self.batches = []
        self.sorted_indexes = []

        for i in data_source:
            label_length = len(i["labels"])
            input_length = len(i["input_ids"])
            length = max(label_length, input_length)
            self.lengths.append(length)

        #This call is not used but is needed to give correct __len__ value
        self._get_batches()

    def __iter__(self):
        self._get_batches()
        # Randomly shuffles order of the batches each epoch
        np.random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def calculate_sorted_indexes(self):
        #Adds a small random number to the lengths, used to randomise order when length is the same
        #Simulates a shuffle without compromising the memory efficiency of dynamic length-based batching
        random_tie_breaking_noise = np.random.rand(len(self.lengths)) * 0.1
        self.sorted_indexes = np.argsort(self.lengths + random_tie_breaking_noise)[::-1]

    def _get_batches(self):
        self.calculate_sorted_indexes()
        self.batches = []
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
