import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ocean_grid_data import OceanGridData
from point_generator import PointGenerator
from ocean_point_array import DisDataGenerator, OceanPointArray

class Sampler:
    def __init__(self, sampling_crop_shape: list[int], downsampling_rate: list[int], n_sampling_point: int, point_generator: PointGenerator) -> None:
        self.ht, self.hz, self.hx = sampling_crop_shape
        self.dt, self.dz, self.dx = downsampling_rate
        self.lt = self.ht // self.dt
        self.lz = self.hz // self.dz
        self.lx = self.hx // self.dx
        self.n_sampling_point = n_sampling_point
        self.point_generator = point_generator
        self.lres_crop_shape = (self.lt, self.lz, self.lx)

    def _get_interp_fun(self, hres_crop: OceanGridData):
        data = hres_crop.data
        return RegularGridInterpolator(
            (np.arange(self.ht), np.arange(self.hz), np.arange(self.hx)), 
            values=data.transpose(2, 3, 4, 0, 1).astype(np.float64), 
            method="linear"
        )

    def down_sample(self, hres_crop: OceanGridData):
        interp_fun = self._get_interp_fun(hres_crop)
        lres_coord = np.stack(np.meshgrid(
            np.linspace(0, self.ht - 1, self.lt),
            np.linspace(0, self.hz - 1, self.lz),
            np.linspace(0, self.hx - 1, self.lx),
            indexing="ij"), axis=-1
        )
        data = interp_fun(lres_coord).transpose(3, 4, 0, 1, 2)
        return OceanGridData(data)
    
    def point_generate(self, hres_crop: OceanGridData) -> np.ndarray:
        grid_shape = [hres_crop.nt, hres_crop.nz, hres_crop.nx]
        return self.point_generator(grid_shape, self.n_sampling_point, hres_crop.sod)

    def point_sample(self, hres_crop: OceanGridData) -> OceanPointArray:
        point_coord = self.point_generate(hres_crop)
        interp_fun = self._get_interp_fun(hres_crop)
        point_value = interp_fun(point_coord).reshape(self.n_sampling_point, -1)
        dis_data_generator = DisDataGenerator(hres_crop)
        point_array = OceanPointArray(point_coord, point_value, dis_data_generator)
        return point_array
