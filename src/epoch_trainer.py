from torch import optim

from recorder import TrainRecorder
from trainer import Trainer
from evaler import Evaler

class EpochTrainer:
    def __init__(self, trainer: Trainer, evaler: Evaler, epoch_num: int, scheduler: optim.lr_scheduler) -> None:
        self.recorder = TrainRecorder()
        self.model = trainer.model
        self.trainer = trainer
        self.evaler = evaler
        self.scheduler = scheduler
        self.optimizer = trainer.optimizer
        self.epoch_num = epoch_num
        self.epoch_index = 1

    def load(self, resume_folder, model_name):
        pass

    def run(self):
        while self.epoch_index <= self.epoch_num:
            self.recorder.logger.info(f"epoch index: {self.epoch_index}")
            self.recorder.metric.new_epoch()
            self.trainer.run()
            self.evaler.run()
            self.recorder.metric.update()
            reg_loss, pde_loss, sum_loss = self.recorder.metric.get_latest_average_loss()
            psnr = self.recorder.metric.get_latest_average_psnr()
            ssim = self.recorder.metric.get_latest_average_ssim()
            self.recorder.logger.info_eopch_metric(self.epoch_index, reg_loss, pde_loss, sum_loss, psnr, ssim)
            self.scheduler.step(sum_loss)
            best_info = {
                metric_name: self.recorder.metric.is_best_metric(metric_name) for metric_name in ["sum_loss", "psnr", "ssim"]
            }
            self.recorder.save_model_scheduler(self.model, self.scheduler, self.epoch_index, best_info)
            self.epoch_index += 1
