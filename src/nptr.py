from abc import ABC, abstractmethod

from ocean_grid_data import OceanGridData
from grid_data_process import TopoMiddled

class Nptr(ABC):
    @abstractmethod
    def __call__(self, isw_data: OceanGridData) -> OceanGridData:
        pass

class GlobelNptr(Nptr):
    def __init__(self) -> None:
        self._topo_middled = TopoMiddled()

    def __call__(self, isw_data: OceanGridData) -> OceanGridData:
        for dataset_index in range(isw_data.nd):
            sod = isw_data.sod[dataset_index, 0, 0]
            self._topo_middled.set_dim2_sod(sod)
            for phy_fea_index, data in enumerate(isw_data.phy_fea_data[dataset_index]):
                data = isw_data.phy_fea_data[dataset_index, phy_fea_index]
                data = self._topo_middled(data)
                isw_data.phy_fea_data[dataset_index, phy_fea_index] = data
        return isw_data

class EageNptr(Nptr):
    def __call__(self, isw_data: OceanGridData) -> OceanGridData:
        raise "the method not completed"

class NptrFactory:
    def __call__(self, is_use, name: str) -> Nptr:
        if is_use:
            if name == "globel_nptr":
                return GlobelNptr()
            elif name == "eage_nptr":
                return EageNptr()
            else:
                raise f"no nptr named {name}"
        else:
            return lambda data: data
