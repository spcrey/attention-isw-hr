from abc import ABC, abstractmethod

import numpy as np
import cv2
from scipy.signal import convolve2d

class ShapeChecker(ABC):
    @abstractmethod
    def __call__(self, data: np.ndarray) -> None:
        pass

class NoneShapeChecker(ABC):
    def __call__(self, data: np.ndarray) -> None:
        pass

class Dim2DataShapeChecker(ShapeChecker):
    def __call__(self, data: np.ndarray) -> None:
        if not len(data.shape) == 2:
            raise "data's shape must be 2D"

class Dim3DataShapeChecker(ShapeChecker):
    def __call__(self, data: np.ndarray) -> None:
        if not len(data.shape) == 3:
            raise "data's shape must be 3D"

class GridDataProcess(ABC):
    _input_data_shape_checker = NoneShapeChecker()
    _output_data_shape_checker = NoneShapeChecker()

    @abstractmethod
    def _process(self, data: np.ndarray):
        pass

    def __call__(self, data: np.ndarray) -> np.ndarray:
        self._input_data_shape_checker(data)
        data = self._process(data)
        self._output_data_shape_checker(data)
        return data

class Dim2ToDim2(GridDataProcess):
    _input_data_shape_checker = Dim2DataShapeChecker()
    _output_data_shape_checker = Dim2DataShapeChecker()

class Dim2ToDim3(GridDataProcess):
    _input_data_shape_checker = Dim2DataShapeChecker()
    _output_data_shape_checker = Dim3DataShapeChecker()

class Dim3ToDim3(GridDataProcess):
    _input_data_shape_checker = Dim3DataShapeChecker()
    _output_data_shape_checker = Dim3DataShapeChecker()

class SodToEage(Dim2ToDim2):
    def _process(self, sod: np.ndarray):
        # sod's shape: [z, x]
        dtype = sod.dtype
        sod_uint8 = (sod[:] * 255).astype(np.uint8)
        sod_uint8 = np.stack([sod_uint8] * 3, axis=-1)
        eage_uint8 = cv2.Canny(sod_uint8, 100, 200)
        eage = np.array(eage_uint8).astype(dtype) / 255
        return eage
    
class EageExpand(Dim2ToDim2):
    def __init__(self, expand_level: int) -> None:
        self._expand_level = expand_level

    def _process(self, eage: np.ndarray):
        # eage's shape: [z, x]
        dtype = eage.dtype
        npdata_expan_soeage = eage[:]
        kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        for _ in range(self._expand_level):
            npdata_expan_soeage = convolve2d(npdata_expan_soeage, kernel, mode="same")
            npdata_expan_soeage[npdata_expan_soeage!=0.0] = 1.0
        npdata_expan_soeage = npdata_expan_soeage.astype(dtype)
        return npdata_expan_soeage

class AddDimT(Dim2ToDim3):
    def __init__(self, nt: int) -> None:
        self._nt = nt

    def _process(self, zx_data: np.ndarray) -> np.ndarray:
        # zx_data's shape: [z, x], tzx_data's shape: [t, z, x]
        dtype = zx_data.dtype
        tzx_npdata = []
        for _ in range(self._nt):
            tzx_npdata.append(zx_data[:])
        tzx_npdata = np.stack(tzx_npdata, 0)
        tzx_npdata = tzx_npdata.astype(dtype)
        return tzx_npdata

class TopoMiddled(Dim3ToDim3):
    _dim2_data_shape_checker = Dim2DataShapeChecker()

    def set_dim2_sod(self, dim2_sod: np.ndarray):
        self._dim2_data_shape_checker(dim2_sod)
        self._dim2_sod = dim2_sod

    def _process(self, dim3_data: np.ndarray):
        if self._dim2_sod.all() == None:
            raise "need to bind sod"
        dtype = dim3_data.dtype
        grid_process_fun = AddDimT(dim3_data.shape[0])
        dim3_sod = grid_process_fun(self._dim2_sod)
        aver = np.sum(dim3_data * dim3_sod) / np.sum(dim3_sod)
        adjsut_dim3_data = dim3_data[:]
        adjsut_dim3_data[dim3_sod==0] = aver
        adjsut_dim3_data = adjsut_dim3_data.astype(dtype)
        return adjsut_dim3_data
