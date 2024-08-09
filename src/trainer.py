import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from cuda_manager import CudaManager
from triple_model import TripleModel
from recorder import TrainRecorder

class Trainer:
    def __init__(self, dataloader: DataLoader, model: TripleModel, loss_fun: nn.modules.loss, optimizer: optim, pde_loss_alpha: float) -> None:
        self.recorder = TrainRecorder()
        self.cuda_manager = CudaManager()
        self.model = model
        self.dataloader = dataloader
        self.loss_fun = loss_fun
        self.optimizer = optimizer
        self.reg_loss_alpha = 1 - pde_loss_alpha
        self.pde_loss_alpha = pde_loss_alpha

    def run(self):
        self.model.train()
        self.model.set_calc_pde(True)
        self.model.set_point_iterate(False)
        for batch_data in self.dataloader:
            lres_crop, point_coord, truth_point_value, point_sod, point_dis = self.adapt(batch_data)
            predict_point_value, pde_value = self.model(lres_crop, point_coord, point_sod, point_dis)
            reg_loss = self.loss_fun(truth_point_value, predict_point_value)
            pde_loss = self.loss_fun(pde_value, torch.zeros_like(pde_value))
            sum_loss = self.reg_loss_alpha * reg_loss + self.pde_loss_alpha * pde_loss
            self.optimizer.zero_grad()
            sum_loss.backward()
            self.optimizer.step()
            self.recorder.metric.add_loss(reg_loss.item(), pde_loss.item(), sum_loss.item())
    
    def adapt(self, batch_data: list[torch.Tensor]):
        dtype = torch.float32
        return [item.to(self.cuda_manager.device).to(dtype) for item in batch_data]
