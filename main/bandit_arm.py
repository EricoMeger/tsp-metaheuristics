import random


class BanditArm:
    __slots__ = ['action_idx', 'successes', 'failures', 'visits']
    
    def __init__(self, action_idx):
        self.action_idx = action_idx
        self.successes = 1.0
        self.failures = 1.0
        self.visits = 0

    def display_info(self, action_name):
        name = str(action_name)
        print(f"  {name:15s}: visits={self.visits:5d}, successes={self.successes:.2f}, "
              f"failures={self.failures:.2f}")

    def sample_thompson(self):
        return random.betavariate(self.successes, self.failures)
