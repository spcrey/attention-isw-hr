import sys

import numpy as np

sys.path.append("src")

from grid_data_process import SodToEage, EageExpand
from point_generator import RandomPointGenerator, EagePointGenerator

class EageGenerator:
    def __init__(self) -> None:
        self.sod_to_eage = SodToEage()
        self.eage_expand = EageExpand(20)

    def __call__(self, sod: np.ndarray) -> np.ndarray:
        eage = self.sod_to_eage(sod)
        eage = self.eage_expand(eage)
        return eage

class EagePointGeneratorV1(EagePointGenerator):
    def __init__(self, random_scale: float, eage_alpha: float) -> None:
        super().__init__(random_scale, eage_alpha)
        self.eage_generator = EageGenerator()

    def exclude_exterior_point(self, coord: np.ndarray, nz: int, nx: int) -> np.ndarray:
        mask1 = coord[:,0] > 0.0
        mask2 = coord[:,1] > 0.0
        mask3 = coord[:,0] < nz - 1
        mask4 = coord[:,1] < nx - 1
        mask = mask1 & mask2 & mask3 & mask4
        coord = coord[mask]
        return coord
    
    def exclude_redundant_point(self, coord, point_num) -> np.ndarray:
        coord_num = coord.shape[0]
        coord = coord[:int(point_num)] if coord_num > point_num else coord
        return coord
    
    def supple_random_point(self, eage_coord, grid_shape, point_num) -> np.ndarray:
        eage_coord_num = len(eage_coord)
        random_coord_num = point_num - eage_coord_num
        point_generator = RandomPointGenerator()
        random_coord = point_generator(grid_shape, random_coord_num, None)
        point_coord = np.concatenate([eage_coord, random_coord], axis=0)
        return point_coord
    
    def add_random_t(self, coord, nt):
        point_num = len(coord)
        coord_t = np.random.rand(point_num, 1) * (nt - 1)
        coord = np.concatenate([coord_t, coord], axis=-1)
        return coord
    
    def get_eage_coord(self, sod):
        sod = sod[0, 0, 0]
        eage = self.eage_generator(sod)
        eage_coord = np.argwhere(eage==1)
        return eage_coord
    
    def adjust_value(self, eage_coord):
        random_adjust_value = np.random.normal(loc=0.0, scale=self.random_scale, size=eage_coord.shape)
        eage_coord = eage_coord + random_adjust_value
        return eage_coord

    def __call__(self, grid_shape: list[int], point_num: int, sod: np.ndarray) -> np.ndarray:
        nt, nz, nx = grid_shape
        eage_coord = self.get_eage_coord(sod)
        eage_coord = self.adjust_value(eage_coord)
        eage_coord = self.exclude_exterior_point(eage_coord, nz, nx)
        eage_coord = self.exclude_redundant_point(eage_coord, point_num)
        eage_coord = self.add_random_t(eage_coord, nt)
        coord = self.supple_random_point(eage_coord, grid_shape, point_num)
        np.random.shuffle(coord)
        return coord
