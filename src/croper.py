import numpy as np

from ocean_grid_data import OceanGridData

# used for dividing data into chunks, for example, dividing a 3x3 data into 4 chunks of 2x2
# 
#  1  2  3    ->  1  2  ,  2  3  ,  4  5  ,  5  6
#  4  5  6        4  5     5  6     7  8     8  9
#  7  8  9

class Croper:
    def __init__(self, full_shape: list[int], crop_shape: list[int]) -> None:
        self.full_nt, self.full_nz, self.full_nx = full_shape
        self.crop_nt, self.crop_nz, self.crop_nx = crop_shape
        self.shape = crop_shape
        dim_start_ids = [
            np.arange(0, self.full_nt - self.crop_nt + 1),
            np.arange(0, self.full_nz - self.crop_nz + 1),
            np.arange(0, self.full_nx - self.crop_nx + 1),
        ] # dst
        self._dst_table = np.stack(np.meshgrid(*dim_start_ids, indexing="ij"), axis=-1).reshape([-1, 3])
        self.full_shape = full_shape

    def __len__(self):
        return len(self._dst_table)
         
    def __call__(self, grid_data: OceanGridData, index: int) -> OceanGridData:
        data = grid_data.data
        data_index = index % grid_data.nd
        table_index = index // grid_data.nd
        t0, z0, x0 = self._dst_table[table_index]
        sliecs = [
            slice(data_index, data_index + 1), slice(None, None), 
            slice(t0, t0 + self.crop_nt), slice(z0, z0 + self.crop_nz), slice(x0, x0 + self.crop_nx)
        ]
        crop = data[*sliecs]
        return OceanGridData(crop)
