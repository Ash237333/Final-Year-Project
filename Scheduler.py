class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, embedding_dim):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.embedding_dim = embedding_dim
        self.step = 0
        
    def update_lr(self):
        self.step += 1
        decay_term = self.step ** (-0.5)
        warmup_term = self.step*self.warmup_steps ** (-1.5)
        lr = (self.embedding_dim**(-0.5)) * min(decay_term, warmup_term)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def step_and_update(self):
        lr =self.update_lr()
        self.optimizer.step()
        return lr, self.step

    def zero_grad(self):
        self.optimizer.zero_grad()
        
    def state_dict(self):
        """
        Save the state of the scheduler, including the optimizer's state and scheduler-specific parameters.
        """
        return {
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step': self.step,
            'warmup_steps': self.warmup_steps,
            'embedding_dim': self.embedding_dim
        }

    def load_state_dict(self, state_dict):
        """
        Load the state of the scheduler, including the optimizer's state and scheduler-specific parameters.
        """
        self.optimizer.load_state_dict(state_dict['optimizer_state_dict'])
        self.step = state_dict['step']
        self.warmup_steps = state_dict['warmup_steps']
        self.embedding_dim = state_dict['embedding_dim']
    