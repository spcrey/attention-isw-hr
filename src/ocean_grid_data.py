import os
import numpy as np
from einops import repeat

class OceanGridData:
    def __init__(self, data) -> None:
        self.nd, self.nc, self.nt, self.nz, self.nx = data.shape
        self.data = data

    def normalize(self, normalizer):
        return normalizer(self.phy_fea_data)
    
    def set_phy_fea_data(self, phy_fea_data: np.ndarray):
        self.data = np.concatenate([phy_fea_data, self.data[:, -4:]], axis=1)

    @property
    def phy_fea_data(self) -> np.ndarray:
        return self.data[:, :-4]
    
    @property
    def sod(self) -> np.ndarray:
        return self.data[:, -1:]

    @property
    def grid(self) -> np.ndarray:
        return self.data[:, -4:-1]

class DataFileLoader:
    def __call__(self, file_folder: str, file_names: list[str], phy_fea_names: list[str], data_shape: list[str]) -> OceanGridData:
        data = []
        for file_name in file_names:
            npdict = np.load(os.path.join(file_folder, file_name))
            data_d = []
            for fea_name in phy_fea_names:
                data_d.append(npdict[fea_name])
            nt = len(data_d[0])
            for i in range(3):
                data_d.append(npdict["grid"][..., i])
            pattern = "z x -> t z x"
            sod = npdict["sod"]
            sod = repeat(sod, pattern, t = nt)
            data_d.append(sod)
            data_d = np.stack(data_d, axis=0)
            data.append(data_d)
        data = np.stack(data, axis=0)
        data = data[:, :, :data_shape[0], :data_shape[1], :data_shape[2]]
        return OceanGridData(data)
