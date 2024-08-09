import sys

sys.path.append("src")

from croper import Croper
from triple_model import TripleModel
from grid_model_factory import GridModelFactory
from ocean_dataset import FullEvalDataset
from logger import Logger
from nptr import NptrFactory
from pde_model import PdeModel
from recorder import EvalRecorder
from cuda_manager import CudaManager
from param_receiver import EvalParamReceiver
from evaler import FullEvaler
from metric import EvalMetric
from point_generator import GridPointGenerator
from sampler import Sampler
from point_model_factory import PointModelFactory

def main():
    param_receiver = EvalParamReceiver()
    param = param_receiver()
    # cuda
    CudaManager(param.use_cuda, param.cuda_devices)
    # recorder(logger and metric)
    logger = Logger("eval", param.log_folder)
    metric = EvalMetric(param.phy_fea_names, param.log_folder)
    recorder = EvalRecorder(logger, metric)
    recorder.logger.info("eval start!")
    # dataset
    nptr = NptrFactory()(True, "globel_nptr")
    point_generator = GridPointGenerator()
    sampler = Sampler(param.data_shape, [4, 8, 8], None, point_generator)
    croper = Croper(param.data_shape, param.sampling_crop_shape)
    nptr_factory = NptrFactory()
    nptr = nptr_factory(param.use_nptr, param.nptr_name)
    dataset = FullEvalDataset("data", "isw_dataset_c307.npz", ["p", "b", "u", "w"], nptr, sampler, croper)
    # model
    factory = GridModelFactory()
    grid_model = factory(param.grid_model_name, len(param.phy_fea_names), param.n_latent_fea, sampler.lres_crop_shape, param.n_grid_model_baselayer_fea, param.n_grid_model_layer)
    factory = PointModelFactory()
    point_model = factory(param.point_model_name, param.n_latent_fea, len(param.phy_fea_names), param.n_point_model_baselayer_fea, param.n_grid_model_layer)
    pde_model = PdeModel(param.phy_fea_names, param.equation_file_path, dataset)
    dual_model = TripleModel(grid_model, point_model, pde_model)
    # eval and run
    evaler = FullEvaler(dataset, dual_model, param.eval_point_batch_size)
    evaler.run()

if __name__ == "__main__":
    main()
