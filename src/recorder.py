from abc import ABC
import argparse
import json
import os
import shutil
from glob import glob
from typing import Optional

import torch
from torch.optim import lr_scheduler

from triple_model import TripleModel
from logger import Logger
from metric import Metric, TrainMetric, EvalMetric
from singleton import singleton

class Recorder(ABC):
    def recorde_param(self, param: argparse.Namespace):
        with open(os.path.join(self.file_folder, "param.json"), "w") as file:
            json.dump(param.__dict__, file, indent=4)
        
@singleton
class TrainRecorder(Recorder):
    def __init__(self, logger: Logger, metric: TrainMetric) -> None:
        self.logger = logger
        self.metric = metric
        self.file_folder = logger.file_folder
    
    def param_backup(self, param):
        with open(os.path.join(self.file_folder, "param.json"), "w") as file:
            json.dump(param.__dict__, file, indent=4) 

    def program_backup(self):
        file_paths = []
        file_paths += glob(os.path.join("*.py"))
        file_paths += glob(os.path.join("src", "*.py"))
        file_paths += glob(os.path.join("approach", "*.py"))
        os.makedirs(os.path.join(self.file_folder, "program"), exist_ok=True)
        os.makedirs(os.path.join(self.file_folder, "program", "src"), exist_ok=True)
        os.makedirs(os.path.join(self.file_folder, "program", "approach"), exist_ok=True)
        for file_path in file_paths:
            shutil.copyfile(file_path, os.path.join(self.file_folder, "program", file_path))

    def save_model_scheduler(self, model: TripleModel, scheduler: lr_scheduler, epoch_index: str, best_info: Optional[dict[bool]]={
        "sum_loss": False, "psnr": False, "ssim": False
    }) -> None:
        state_dict = {
            "model": model.state_dict(),
            "optimizer": scheduler.optimizer.state_dict(),
        }
        model_name = "epoch_" + str(epoch_index).rjust(3, "0")
        file_name = "checkpoint_" + model_name + ".pth.tar"
        torch.save(state_dict, os.path.join(self.file_folder, file_name))
        last_model_name = "epoch_" + str(epoch_index - 1).rjust(3, "0")
        last_file_name = "checkpoint_" + last_model_name + ".pth.tar"
        if os.path.exists(os.path.join(self.file_folder, last_file_name)):
            os.remove(os.path.join(self.file_folder, last_file_name))
        for metric_name in best_info.keys():
            if best_info[metric_name]:
                copy_model_name = "best_" + metric_name
                copy_file_name =  "checkpoint_" + copy_model_name + ".pth.tar"
                shutil.copy(os.path.join(self.file_folder, file_name), os.path.join(self.file_folder, copy_file_name))

    def load_model_scheduler(self, model_path: str, model: TripleModel, scheduler: lr_scheduler):
        if model_path:
            state_dict = torch.load(model_path)
            model.load_state_dict(state_dict["model"])
            scheduler.optimizer.load_state_dict(state_dict["optimizer"])
            return model, scheduler
        else:
            model, scheduler

@singleton
class EvalRecorder(Recorder):
    def __init__(self, logger: Logger, metric: EvalMetric) -> None:
        self.logger = logger
        self.metric = metric
        self.file_folder = logger.file_folder

    def load_model(model_path: str, model: TripleModel):
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict["model"])
        return model

@singleton
class VisualRecorder(Recorder):
    def __init__(self, logger: Logger) -> None:
        self.logger = logger
        self.file_folder = logger.file_folder

def test_train_recorder():
    from param_receiver import TrainParamReceiver

    recorder = TrainRecorder(["p", "b", "u", "w"], os.path.join("log", "recorder_test"))
    recorder.logger.info("Hello ")
    recorder.program_backup()
    param_receiver = TrainParamReceiver()
    param = param_receiver()
    recorder.recorde_param(param)

def test_eval_recorder():
    from param_receiver import EvalParamReceiver

    recorder = EvalRecorder(["p", "b", "u", "w"], os.path.join("log", "recorder_test", "eval"))
    recorder.logger.info("Hello ")
    param_receiver = EvalParamReceiver()
    param = param_receiver()
    recorder.recorde_param(param)

def main():
    test_eval_recorder()

if __name__ == "__main__":
    main()
