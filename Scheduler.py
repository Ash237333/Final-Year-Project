import torch

class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, embedding_dim):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.embedding_dim = embedding_dim
        
    def update_lr(self, step_num):
        decay_term = step_num ** -1.5
        warmup_term = step_num*self.warmup_steps ** -1.5
        lr = (self.embedding_dim**-0.5) * min(decay_term, warmup_term)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    