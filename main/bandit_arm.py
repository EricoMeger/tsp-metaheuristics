import random


class BanditArm:
    __slots__ = ['action_idx', 'successes', 'failures', 'visits']
    
    def __init__(self, action_idx):
        self.action_idx = action_idx
        self.successes = 1.0
        self.failures = 1.0
        self.visits = 0

    def sample_thompson(self):
        return random.betavariate(self.successes, self.failures)
