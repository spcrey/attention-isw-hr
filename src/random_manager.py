import numpy as np
import torch

from singleton import singleton

@singleton
class RandomManager:
    def __init__(self, seed: int) -> None:
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
