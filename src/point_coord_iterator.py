from abc import ABC, abstractmethod

import numpy as np

class Iterator(ABC):
    @abstractmethod
    def has_next(self):
        pass

    @abstractmethod
    def next(self):
        pass

class PointCoordIterator(Iterator):
    def __init__(self, point_coord: np.ndarray, batch_size: int):
        self.point_coord = point_coord
        self.batch_size = batch_size
        self.index = 0

    def has_next(self):
        return self.index < self.point_coord.shape[-2]

    def next(self):
        end = min(self.index + self.batch_size, self.point_coord.shape[-2])
        item = self.point_coord[:, self.index: end]
        self.index = end
        return item
