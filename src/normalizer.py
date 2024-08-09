import numpy as np
import torch

class Normalizer:
    def __init__(self, grid_data: np.ndarray, crop_shape: list[int]) -> None:
        self.grid_data = grid_data
        self.mean, self.std = self.calc_mean_std()
        self.crop_shape = crop_shape

    def calc_mean_std(self):
        mean = np.mean(self.grid_data, axis=(0, 2, 3, 4))
        std = np.std(self.grid_data, axis=(0, 2, 3, 4))
        return mean, std

    def normalize_grid_data(self, grid_data: np.ndarray) -> np.ndarray:
        return (grid_data - self.mean.reshape(1,-1,1,1,1)) / self.std.reshape(1,-1,1,1,1)
    
    def normalize_point_coord(self, point_coord):
        return point_coord / np.array(self.crop_shape).reshape(1, -1)
    
    def normalize_point_value(self, grid_data: np.ndarray) -> np.ndarray:
        return (grid_data - self.mean.reshape(1,-1)) / self.std.reshape(1,-1)

    def denormalize_grid_data(self, grid_data: np.ndarray) -> np.ndarray:
        return grid_data * self.std.reshape(1, -1, 1, 1, 1) + self.mean.reshape(1, -1, 1, 1, 1)

    def denormalize_point_value(self, point_value: np.ndarray) -> np.ndarray:
        return point_value
