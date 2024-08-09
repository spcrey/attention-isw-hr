from abc import ABC, abstractmethod

import numpy as np
import torch

class PointGenerator(ABC):
    @abstractmethod
    def __call__(self, grid_shape: list[int], point_num: int, sod: np.ndarray) -> np.ndarray:
        pass

class RandomPointGenerator(PointGenerator):
    def __call__(self, grid_shape: list[int], point_num: int, sod: np.ndarray) -> np.ndarray:
        point_coord = np.random.rand(point_num, 3) * (np.array(grid_shape) - 1)
        return point_coord
    
class GridPointGenerator(PointGenerator):
    def __init__(self, eps=1e-6) -> None:
        self.eps = eps

    def __call__(self, grid_shape: list[int], point_num, sod: np.ndarray) -> np.ndarray:
        t_coord = np.linspace(grid_shape[0] * self.eps, grid_shape[0] * (1 - self.eps), grid_shape[0])
        z_coord = np.linspace(grid_shape[1] * self.eps, grid_shape[1] * (1 - self.eps), grid_shape[1])
        x_coord = np.linspace(grid_shape[2] * self.eps, grid_shape[2] * (1 - self.eps), grid_shape[2])
        coord = np.stack(np.meshgrid(t_coord, z_coord, x_coord, indexing="ij"), axis=-1)
        coord = coord.reshape([-1, 3])
        return coord

class EagePointGenerator(PointGenerator):
    def __init__(self, random_scale: float, eage_alpha: float) -> None:
        self.random_scale = random_scale
        self.eage_alpha = eage_alpha

    @abstractmethod
    def __call__(self, grid_shape: list[int], point_num: int, sod: np.ndarray) -> np.ndarray:
        pass

def main():
    point_generator = GridPointGenerator()
    point_coord = point_generator([16, 128, 128], None, None)
    pass
    

if __name__ == "__main__":
    main()
