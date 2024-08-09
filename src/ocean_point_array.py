import numpy as np

from ocean_grid_data import OceanGridData

class DisDataGenerator:
    def __init__(self, crop_data: OceanGridData) -> None:
        grid = crop_data.grid
        self.grid_t = grid[0, 0, :, 0, 0]
        self.grid_z = grid[0, 1, 0, :, 0]
        self.grid_x = grid[0, 2, 0, 0, :]

    def __call__(self, point_grid) -> np.ndarray:
        point_dis = np.zeros_like(point_grid)
        for point_id in range(len(point_grid)):
            grid_t, grid_z, grid_x = point_grid[point_id]
            closest_index = np.abs(self.grid_t - grid_t).argmin()
            if self.grid_t[closest_index] < grid_t:
                t0 = closest_index
            else:
                t0 = closest_index - 1
            t0 = min(t0, len(self.grid_t) - 2)
            closest_index = np.abs(self.grid_z - grid_z).argmin()
            if self.grid_z[closest_index] < grid_z:
                z0 = closest_index
            else:
                z0 = closest_index - 1
            z0 = min(z0, len(self.grid_z) - 2)
            closest_index = np.abs(self.grid_x - grid_x).argmin()
            if self.grid_x[closest_index] < grid_x:
                x0 = closest_index
            else:
                x0 = closest_index - 1
            x0 = min(x0, len(self.grid_x) - 2)
            point_dis[point_id][0] = np.abs(self.grid_t[t0 + 1] - self.grid_t[t0])
            point_dis[point_id][1] = np.abs(self.grid_z[z0 + 1] - self.grid_z[z0])
            point_dis[point_id][2] = np.abs(self.grid_x[x0 + 1] - self.grid_x[x0])
        return point_dis

class OceanPointArray:
    def __init__(self, coord: np.ndarray, value: np.ndarray, dis_data_generator: DisDataGenerator) -> None:
        self.coord = coord
        self.value = value
        self.dis_data_generator = dis_data_generator

    def normalize(self, normalizer):
        return normalizer(self.phy_fea_data)
    
    @property
    def phy_fea_data(self) -> np.ndarray:
        return self.value[:, :-4]
    
    @property
    def dis_data(self) -> np.ndarray:
        return self.dis_data_generator(self.value[:, -4:-1])
    
    @property
    def sod_data(self) -> np.ndarray:
        return self.value[:, -1:]
