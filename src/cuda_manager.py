import torch
from torch import nn

from singleton import singleton

@singleton
class CudaManager:
    def __init__(self, use_cuda: bool, cuda_devices: list[int], batch_size_per_cuda=None) -> None:
        self.cuda_devices = cuda_devices if use_cuda else []
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.use_cuda = use_cuda
        self.cuda_num = len(self.cuda_devices)
        self.batch_size = batch_size_per_cuda * self.cuda_num if batch_size_per_cuda else None

    @property
    def dataloader_args(self) -> dict:
        if self.use_cuda:
            return {
                "num_workers": self.cuda_num, 
                "pin_memory": True
            }
        else:
            return {}
        
    def model_cuda_adapt(self, model: nn.Module) -> nn.Module | nn.DataParallel[nn.Module]:
        if self.use_cuda:
            return nn.DataParallel(model, self.cuda_devices).to(self.device)
        else:
            return model
        
    def model_parallel_adapt(self, model: nn.Module) -> nn.Module | nn.DataParallel[nn.Module]:
        if self.cuda_num > 1:
            return nn.DataParallel(model, self.cuda_devices).to(self.device)
        else:
            return model
