from torch.optim import Optimizer

class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, embedding_dim):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.embedding_dim = embedding_dim
        self.step = 0
        
    def update_lr(self):
        self.step += 1
        decay_term = self.step ** (-1.5)
        warmup_term = self.step*self.warmup_steps ** (-1.5)
        lr = (self.embedding_dim**(-0.5)) * min(decay_term, warmup_term)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def step_and_update(self):
        lr =self.update_lr()
        self.optimizer.step()
        return lr

    def zero_grad(self):
        self.optimizer.zero_grad()
    