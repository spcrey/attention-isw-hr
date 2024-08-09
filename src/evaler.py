from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader

from cuda_manager import CudaManager
from triple_model import TripleModel
from metric import BatchFeaMetricCalculator, UseNorm, PsnrStrategy, SsimStrategy
from recorder import TrainRecorder, EvalRecorder

class Evaler(ABC):
    def __init__(self) -> None:
        self.batch_fea_calculator = BatchFeaMetricCalculator()
        self.batch_fea_calculator.set_norm_strategy(UseNorm())

    def calc_fea_psnr(self, truth_hres_crop, predict_hres_crop):
        self.batch_fea_calculator.set_metric_strategy(PsnrStrategy())
        return self.batch_fea_calculator.calc(truth_hres_crop, predict_hres_crop)
    
    def calc_fea_ssim(self, truth_hres_crop, predict_hres_crop):
        self.batch_fea_calculator.set_metric_strategy(SsimStrategy())
        return self.batch_fea_calculator.calc(truth_hres_crop, predict_hres_crop)

    @abstractmethod
    def run(self):
        pass

class CropEvaler(Evaler):
    def __init__(self, dataloader: DataLoader, model: TripleModel, eval_point_batch_size: int) -> None:
        super().__init__()
        self.cuda_manager = CudaManager()
        self.recorder = TrainRecorder()
        self.dataloader = dataloader
        self.model = model
        self.eval_point_batch_size = eval_point_batch_size
        self.normalizer = dataloader.dataset.normalizer

    def run(self):
        self.model.eval()
        self.model.set_calc_pde(False)
        self.model.set_point_iterate(True, self.eval_point_batch_size)
        for batch_data in self.dataloader:
            truth_hres_crop, lres_crop, point_coord = self.adapt(batch_data)
            predict_point_value = self.model(lres_crop, point_coord)
            predict_hres_crop = predict_point_value.reshape(*truth_hres_crop.shape)
            predict_hres_crop = self.normalizer.denormalize_grid_data(predict_hres_crop.cpu().detach().numpy())
            fea_psnr = self.calc_fea_psnr(truth_hres_crop, predict_hres_crop)
            fea_ssim = self.calc_fea_ssim(truth_hres_crop, predict_hres_crop)
            self.recorder.metric.add_psnr_ssim(fea_psnr, fea_ssim)

    def adapt(self, batch_data: list[torch.Tensor]):
        truth_hres_crop, lres_crop, point_coord = batch_data
        dtype = torch.float32
        return [
            truth_hres_crop.numpy(),
            lres_crop.to(self.cuda_manager.device).to(dtype),
            point_coord.to(self.cuda_manager.device).to(dtype),
        ]
            
class FullEvaler(Evaler):
    def __init__(self, dataset, model, eval_point_batch_size) -> None:
        super().__init__()
        self.cuda_manager = CudaManager()
        self.recorder = EvalRecorder()
        self.dataset = dataset
        self.model = model
        self.eval_point_batch_size = eval_point_batch_size
        self.normalizer = dataset.normalizer

    def run(self):
        self.model.eval()
        self.model.set_calc_pde(False)
        self.model.set_point_iterate(True, self.eval_point_batch_size)
        truth_hres_crop, lres_crop, point_coord = self.adapt(self.dataset[0])
        predict_point_value = self.model(lres_crop, point_coord)
        predict_hres_crop = predict_point_value.reshape(*truth_hres_crop.shape)
        predict_hres_crop = self.normalizer.denormalize_grid_data(predict_hres_crop.cpu().detach().numpy())
        fea_psnr = self.calc_fea_psnr(truth_hres_crop, predict_hres_crop)
        fea_ssim = self.calc_fea_ssim(truth_hres_crop, predict_hres_crop)
        self.recorder.metric.add_psnr_ssim(fea_psnr, fea_ssim)
        pass

    def adapt(self, data: list[torch.Tensor]):
        truth_hres_crop, lres_crop, point_coord = data
        dtype = torch.float32
        return [
            truth_hres_crop.reshape(1, *truth_hres_crop.shape),
            torch.from_numpy(lres_crop.reshape(1, *lres_crop.shape)).to(self.cuda_manager.device).to(dtype),
            torch.from_numpy(point_coord.reshape(1, *point_coord.shape)).to(self.cuda_manager.device).to(dtype),
        ]
