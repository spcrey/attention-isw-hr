import sys

sys.path.append("src")

from metric import TrainMetric
from triple_model import TripleModel
from epoch_trainer import EpochTrainer
from ocean_dataset import TrainDataset, CropEvalDataset, TrainDataLoaderFactory, CropEvalDataLoaderFactory
from eage_point_generator_factory import EagePointGeneratorFactory
from logger import Logger
from recorder import TrainRecorder
from sampler import Sampler
from optimizer import OptimizerFactory, Scheduler
from grid_model_factory import GridModelFactory
from point_model_factory import PointModelFactory
from loss_fun import LossFunFactory
from pde_model import PdeModel
from cuda_manager import CudaManager
from random_manager import RandomManager
from param_receiver import TrainParamReceiver
from croper import Croper
from trainer import Trainer
from evaler import CropEvaler
from point_generator import GridPointGenerator
from nptr import NptrFactory

def main():
    param_receiver = TrainParamReceiver()
    param = param_receiver()
    # cuda
    CudaManager(param.use_cuda, param.cuda_devices, param.batch_size_per_cuda)
    # random
    RandomManager(param.seed)
    # recorder(include logger and metric)
    logger = Logger("train", param.log_folder)
    metric = TrainMetric(param.phy_fea_names, param.log_folder)
    recorder = TrainRecorder(logger, metric)
    recorder.logger.info("train start!")
    recorder.program_backup()
    recorder.param_backup(param)
    # eage sampling and numerical preprocess of topography region (nptr)
    croper = Croper(param.data_shape, param.sampling_crop_shape)
    eage_point_generator_factory = EagePointGeneratorFactory()
    eage_point_generator = eage_point_generator_factory(param.use_eage_sampling, param.eage_sampling_name, param.eage_sampling_random_scale, param.eage_sampling_eage_alpha)
    grid_point_generator = GridPointGenerator()
    train_sampler = Sampler(param.sampling_crop_shape, param.downsampling_rate, param.n_sampling_point, eage_point_generator)
    eval_sampler = Sampler(param.sampling_crop_shape, param.downsampling_rate, param.n_sampling_point, grid_point_generator)
    nptr = NptrFactory()(param.use_nptr, param.nptr_name)
    # dataset(train and eval)
    train_dataset = TrainDataset(param.train_dataset_folder, param.train_dataset_names, param.phy_fea_names, nptr, train_sampler, croper)
    train_dataloader = TrainDataLoaderFactory()(train_dataset, param.n_sampling_crop)
    eval_dataset = CropEvalDataset(param.train_dataset_folder, param.train_dataset_names, param.phy_fea_names, nptr, eval_sampler, croper)
    eval_dataloader = CropEvalDataLoaderFactory()(eval_dataset, param.n_eval_sampling_crop)
    # model, loss_fun and scheduler(optimizer)
    factory = GridModelFactory()
    grid_model = factory(param.grid_model_name, len(param.phy_fea_names), param.n_latent_fea, train_sampler.lres_crop_shape, param.n_grid_model_baselayer_fea, param.n_grid_model_layer)
    factory = PointModelFactory()
    point_model = factory(param.point_model_name, param.n_latent_fea, len(param.phy_fea_names), param.n_point_model_baselayer_fea, param.n_grid_model_layer)
    pde_model = PdeModel(param.phy_fea_names, param.equation_file_path, train_dataset)
    dual_model = TripleModel(grid_model, point_model, pde_model)
    loss_fun = LossFunFactory()(param.loss_fun_type)
    optimizer = OptimizerFactory()(param.optimizer_type, dual_model, param.learning_rate)
    scheduler = Scheduler(optimizer)
    # train, load and run
    trainer = Trainer(train_dataloader, dual_model, loss_fun, optimizer, param.pde_loss_alpha)
    evaler = CropEvaler(eval_dataloader, dual_model, param.eval_point_batch_size)
    epoch_trainer = EpochTrainer(trainer, evaler, param.epoch_num, scheduler)
    if param.resume_folder:
        epoch_trainer.load(param.resume_folder, param.resume_model_name)
    epoch_trainer.run()

if __name__ == "__main__":
    main()
