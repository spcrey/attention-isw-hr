from abc import ABC, abstractmethod

from torch.utils.data import DataLoader, RandomSampler

from cuda_manager import CudaManager
from ocean_dataset import TrainDataset, CropEvalDataset

class DataLoaderFoctory(ABC):
    @abstractmethod
    def __call__(self) -> DataLoader:
        pass

class TrainDataLoaderFactory(DataLoaderFoctory):
    def __init__(self, n_sampling_crop: int) -> None:
        cuda_manager = CudaManager()
        self.n_sampling_crop = n_sampling_crop
        self.batch_size = cuda_manager.batch_size
        self.loader_args = cuda_manager.dataloader_args

    def __call__(self, dataset: TrainDataset) -> DataLoader:
        sampler = RandomSampler(dataset, replacement=True, num_samples=self.n_sampling_crop)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=True, sampler=sampler, **self.loader_args)
        return loader

class CropEvalDataLoaderFactory(DataLoaderFoctory):
    def __init__(self, n_sampling_crop: int) -> None:
        cuda_manager = CudaManager()
        self.n_sampling_crop = n_sampling_crop
        self.batch_size = cuda_manager.batch_size
    
    def __call__(self, dataset: CropEvalDataset) -> DataLoader:
        sampler = RandomSampler(dataset, replacement=True, num_samples=self.n_sampling_crop)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, sampler=sampler)
        return loader
