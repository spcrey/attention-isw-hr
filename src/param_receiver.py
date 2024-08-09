from abc import ABC, abstractmethod
import argparse
import json
import os

class ParamReceiver(ABC):
    def __call__(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser()
        self._add_arguments(parser)
        param = parser.parse_args()
        self._update_by_json(param)
        return param

    @abstractmethod
    def _add_arguments(self) -> None:
        pass

    def _update_by_json(self, param: argparse.Namespace) -> argparse.Namespace:
        with open(self._json_file_path, "r") as file:
            json_file_args = json.load(file)
        for key in param.__dict__.keys():
            if param.__dict__[key] == None and not json_file_args[key] == None:
                param.__dict__[key] = json_file_args[key]
        return param

class TrainParamReceiver(ParamReceiver):
    def __init__(self) -> None:
        self._json_file_path = "train_param.json"

    def _add_arguments(self, parser) -> None:
        parser.add_argument("--batch_size_per_cuda", type=int)
        parser.add_argument("--eval_point_batch_size", type=list[int])
        parser.add_argument("--epoch_num", type=int)
        parser.add_argument("--n_sampling_crop", type=int, help="num of sampling crop per epoch")
        parser.add_argument("--n_eval_sampling_crop", type=int, help="num of sampling crop per epoch when eval")
        parser.add_argument("--n_sampling_point", type=int, help="num of sampling crop per sampling crop")
        parser.add_argument("--data_shape", type=list[int], help="shape of full data, include t, z and x")
        parser.add_argument("--sampling_crop_shape", type=list[int], help="shape of sampling crop, include t, z and x")
        parser.add_argument("--learning_rate", type=float)
        parser.add_argument("--seed", type=int, help="seed for numpy and torch")
        parser.add_argument("--log_folder", type=str)
        parser.add_argument("--downsampling_rate", type=list[int], help="downsampling rate of sampling crop, include t, z and x")
        parser.add_argument("--phy_fea_names", type=list[str])
        parser.add_argument("--train_dataset_names", type=list[str])
        parser.add_argument("--train_dataset_folder", type=str)
        parser.add_argument("--n_latent_fea", type=str)
        parser.add_argument("--use_cuda", type=bool)
        parser.add_argument("--cuda_devices", type=list[int])
        parser.add_argument("--loss_fun_type", type=str)
        parser.add_argument("--optimizer_type", type=str)
        parser.add_argument("--equation_file_path", type=str)
        parser.add_argument("--pde_loss_alpha", type=float)
        # resume
        parser.add_argument("--resume_folder", type=str)
        parser.add_argument("--resume_model_name", type=str)
        # model
        parser.add_argument("--grid_model_name", type=str)
        parser.add_argument("--n_grid_model_baselayer_fea", type=int)
        parser.add_argument("--n_grid_model_layer", type=int)
        parser.add_argument("--point_model_name", type=str)
        parser.add_argument("--n_point_model_baselayer_fea", type=int)
        parser.add_argument("--n_point_model_layer", type=int)
        # eage sampling
        parser.add_argument("--use_eage_sampling", type=bool)
        parser.add_argument("--eage_sampling_name", type=str)
        parser.add_argument("--eage_sampling_random_scale", type=float)
        parser.add_argument("--eage_sampling_eage_alpha", type=float)
        # numerical preprocess of topography region (nptr)
        parser.add_argument("--use_nptr", type=bool)
        parser.add_argument("--nptr_name", type=str)

class EvalParamReceiver(ParamReceiver):
    def __init__(self) -> None:
        self._json_file_path = "eval_param.json"

    def _add_arguments(self, parser):
        parser.add_argument("--log_folder", type=str)
        parser.add_argument("--dataset_name", type=str)
        parser.add_argument("--dataset_folder", type=str)
        parser.add_argument("--data_shape", type=list[int])
        parser.add_argument("--eval_folder", type=str)
        parser.add_argument("--use_cuda", type=bool)
        parser.add_argument("--cuda_devices", type=list[int])
        parser.add_argument("--eval_point_batch_size", type=list[int])

    def _update_by_train(self, param):
        with open(os.path.join(param.log_folder, "param.json"), "r") as file:
            train_args = json.load(file)
        inher_keys = [
            "sampling_crop_shape", "phy_fea_names", "downsampling_rate", "n_latent_fea", "grid_model_name", 
            "n_grid_model_baselayer_fea", "n_grid_model_layer", "point_model_name", "n_point_model_baselayer_fea",
            "n_point_model_layer", "use_nptr", "nptr_name", "equation_file_path", 
        ]
        for key in train_args.keys():
            if key in inher_keys:
                param.__dict__[key] = train_args[key]
        return param

    def __call__(self) -> argparse.Namespace:
        param = super().__call__()
        param = self._update_by_train(param)
        return param

class VisualParamReceiver(ParamReceiver):
    def _add_arguments(self, parser):
        parser.add_argument("--eval_folder", type=str)
        parser.add_argument("--visual_folder", type=str)

    def _update_by_eval(self):
        pass
