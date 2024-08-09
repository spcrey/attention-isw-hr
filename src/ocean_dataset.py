from abc import ABC, abstractmethod
import numpy as np
from torch.utils import data

from cuda_manager import CudaManager
from ocean_grid_data import OceanGridData, DataFileLoader
from ocean_point_array import OceanPointArray
from croper import Croper
from sampler import Sampler
from nptr import Nptr
from normalizer import Normalizer

class OceanDataset(data.Dataset):
    def __init__(self, file_folder: str, file_names: list[str], phy_fea_names: list[str], nptr: Nptr, sampler: Sampler, croper: Croper) -> None:
        super(OceanDataset, self).__init__()
        loader = DataFileLoader()
        self.full_data = loader(file_folder, file_names, phy_fea_names, croper.full_shape)
        self.full_data = nptr(self.full_data)
        self.normalizer = Normalizer(self.full_data.phy_fea_data, croper.shape)
        self.croper = croper
        self.sampler = sampler

    def __len__(self):
        return len(self.croper) * self.full_data.nd

    @abstractmethod
    def __getitem__(self, index):
        pass

class TrainDataset(OceanDataset):
    def __init__(self, file_folder: str, file_names: list[str], phy_fea_names: list[str], nptr: Nptr, sampler: Sampler, croper: Croper) -> None:
        super().__init__(file_folder, file_names, phy_fea_names, nptr, sampler, croper)

    def __getitem__(self, index):
        hres_crop = self.croper(self.full_data, index)
        lres_crop = self.sampler.down_sample(hres_crop)
        point_array = self.sampler.point_sample(hres_crop)
        adapter = TrainBatchDataAdapter()
        data = adapter(lres_crop, point_array, self.normalizer)
        return data
    
class CropEvalDataset(OceanDataset):
    def __init__(self, file_folder: str, file_names: list[str], phy_fea_names: list[str], nptr: Nptr, sampler: Sampler, croper: Croper) -> None:
        super().__init__(file_folder, file_names, phy_fea_names, nptr, sampler, croper)
    
    def __getitem__(self, index):
        hres_crop = self.croper(self.full_data, index)
        lres_crop = self.sampler.down_sample(hres_crop)
        point_coord = self.sampler.point_generate(hres_crop)
        adapter = EvalBatchDataAdapter()
        data = adapter(hres_crop, lres_crop, point_coord, self.normalizer)
        return data

class FullEvalDataset(OceanDataset):
    def __init__(self, file_folder: str, file_name: str, phy_fea_names: list[str], nptr: Nptr, sampler: Sampler, croper: Croper) -> None:
        super().__init__(file_folder, [file_name], phy_fea_names, nptr, sampler, croper)

    def __len__(self):
        return 1
    
    def __getitem__(self, index):
        hres_crop = self.full_data
        lres_crop = self.sampler.down_sample(hres_crop)
        point_coord = self.sampler.point_generate(hres_crop)
        adapter = EvalBatchDataAdapter()
        data = adapter(hres_crop, lres_crop, point_coord, self.normalizer)
        return data

class BatchDataAdapter:
    @abstractmethod
    def __call__(self):
        pass
    
class TrainBatchDataAdapter(BatchDataAdapter):
    def __call__(self, lres_crop: OceanGridData, point_array: OceanPointArray, normalizer: Normalizer) -> tuple[np.ndarray]:
        return (
            normalizer.normalize_grid_data(lres_crop.phy_fea_data)[0].astype(np.float32),
            normalizer.normalize_point_coord(point_array.coord).astype(np.float32),
            normalizer.normalize_point_value(point_array.phy_fea_data).astype(np.float32),
            point_array.sod_data.astype(np.float32),
            point_array.dis_data.astype(np.float32),
        )

class EvalBatchDataAdapter(BatchDataAdapter):
    def __call__(self, hres_crop: OceanGridData, lres_crop: OceanGridData, point_coord: np.ndarray, normalizer: Normalizer) -> tuple[np.ndarray]:
        return (
            normalizer.normalize_grid_data(hres_crop.phy_fea_data)[0].astype(np.float32),
            normalizer.normalize_grid_data(lres_crop.phy_fea_data)[0].astype(np.float32),
            normalizer.normalize_point_coord(point_coord).astype(np.float32),
        )

class OceanDataLoader(data.DataLoader):
    def __init__(self, dataset: data.Dataset, batch_size: int, n_sampling_crop: int, drop_last: bool, cuda_args: dict):
        sampler = data.RandomSampler(dataset, replacement=True, num_samples=n_sampling_crop)
        super().__init__(dataset, batch_size, False, sampler, drop_last=drop_last, **cuda_args)

class TrainDataLoader(OceanDataLoader):
    def __init__(self, dataset: data.Dataset, batch_size: int, n_sampling_crop: int, cuda_args):
        super().__init__(dataset, batch_size, n_sampling_crop, True, cuda_args)

class CropEvalDataLoader(OceanDataLoader):
    def __init__(self, dataset: data.Dataset, batch_size: int, n_sampling_crop: int):
        super().__init__(dataset, batch_size, n_sampling_crop, False, {})

class OceanDataLoaderFactory(ABC):
    @abstractmethod
    def __call__(self) -> OceanDataLoader:
        pass

class TrainDataLoaderFactory(OceanDataLoaderFactory):
    def __init__(self) -> None:
        self._cuda_manager = CudaManager()

    def __call__(self, dataset: data.Dataset, n_sampling_crop: int) -> TrainDataLoader:
        batch_size = self._cuda_manager.batch_size
        cuda_args = self._cuda_manager.dataloader_args
        dataloader = TrainDataLoader(dataset, batch_size, n_sampling_crop, cuda_args)
        return dataloader
    
class CropEvalDataLoaderFactory(OceanDataLoaderFactory):
    def __init__(self) -> None:
        self._cuda_manager = CudaManager()

    def __call__(self, dataset: data.Dataset, n_sampling_crop: int) -> CropEvalDataLoader:
        batch_size = self._cuda_manager.batch_size
        dataloader = CropEvalDataLoader(dataset, batch_size, n_sampling_crop)
        return dataloader

def get_test_train_dataset():
    from nptr import NptrFactory
    from eage_point_generator_factory import EagePointGeneratorFactory

    use_nptr = True
    use_eage_sampling = True

    nptr = NptrFactory()(use_nptr, "nptr")
    point_generator_factory = EagePointGeneratorFactory()
    point_generator = point_generator_factory(use_eage_sampling, "eage_sample_v1", 7.0, 0.5)
    sampler = Sampler([16, 128, 128], [4, 8, 8], 512, point_generator)
    croper = Croper([256, 128, 512], [16, 128, 128])
    dataset = TrainDataset("data", ["isw_dataset_c307.npz", "isw_dataset_c307.npz"], ["p", "b", "u", "w"], nptr, sampler, croper)
    return dataset

def test_train_dataset():
    dataset = get_test_train_dataset()
    lres_crop, point_coord, point_value, point_sod, point_dis = dataset[0]
    pass

def test_crop_eval_dataset():
    from nptr import NptrFactory
    from point_generator import GridPointGenerator

    nptr = NptrFactory()(True, "nptr")
    point_generator = GridPointGenerator()
    sampler = Sampler([16, 128, 128], [4, 8, 8], None, point_generator)
    croper = Croper([256, 128, 512], [16, 128, 128])
    dataset = CropEvalDataset("data", ["isw_dataset_c307.npz", "isw_dataset_c307.npz"], ["p", "b", "u", "w"], nptr, sampler, croper)
    hres_crop, lres_crop, point_coord = dataset[0]
    pass

def test_full_eval_dataset():
    from nptr import NptrFactory
    from point_generator import GridPointGenerator
    from point_coord_iterator import PointCoordIterator

    nptr = NptrFactory()(True, "nptr")
    point_generator = GridPointGenerator()
    data_shape = [256, 128, 512]
    sampler = Sampler(data_shape, [4, 8, 8], None, point_generator)
    croper = Croper([256, 128, 512], [16, 128, 128])
    dataset = FullEvalDataset("data", "isw_dataset_c307.npz", ["p", "b", "u", "w"], nptr, sampler, croper)
    hres_crop, lres_crop, point_coord = dataset[0]
    point_coord_iterator = PointCoordIterator(point_coord, 1000)
    point_value = []
    while point_coord_iterator.has_next():
        batch_point_coord = point_coord_iterator.next()
        # process start
        batch_point_value = [
            batch_point_coord[:,0] + batch_point_coord[:,1], 
            batch_point_coord[:,1] - batch_point_coord[:,2],
            batch_point_coord[:,0] - batch_point_coord[:,1], 
            batch_point_coord[:,1] + batch_point_coord[:,2],
        ]
        batch_point_value = np.stack(batch_point_value, axis=-1)
        # process end
        point_value.append(batch_point_value)
    point_value = np.concatenate(point_value, axis=0)
    point_value = point_value.reshape(*hres_crop.shape[1:], -1)
    pass

def main():
    test_full_eval_dataset()

if __name__ == "__main__":
    main()
