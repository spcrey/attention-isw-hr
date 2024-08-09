import os
import json

import torch
from torch.autograd import grad
import sympy
from sympy.parsing.sympy_parser import parse_expr

from ocean_dataset import OceanDataset

class PdeModel:
    def __init__(self, phy_fea_names: list[str], equation_file_path: str, dataset: OceanDataset) -> None:
        self.phy_fea_names = phy_fea_names
        self.dim_names = ["t", "z", "x"]
        self.dis_names = ["dis_t", "dis_z", "dis_x"]
        self.torch_dif = lambda y, x: grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True, allow_unused=True)[0]
        normalizer = dataset.normalizer
        crop_shape = dataset.croper.shape
        with open(equation_file_path, "r") as file:
            equation = json.load(file)
        for equation_name in equation.keys():
            equation[equation_name] = parse_expr(equation[equation_name])
            for phy_fea_id, phy_fea_name in enumerate(phy_fea_names):
                mean = normalizer.mean[phy_fea_id]
                std = normalizer.std[phy_fea_id]
                equation[equation_name] = equation[equation_name].subs(phy_fea_name, f"{phy_fea_name} * {std} + {mean}")
            equation[equation_name] = equation[equation_name].subs("nt", f"1. / (({crop_shape[0] - 1}) * {self.dis_names[0]})")
            equation[equation_name] = equation[equation_name].subs("nz", f"1. / (({crop_shape[1] - 1}) * {self.dis_names[1]})")
            equation[equation_name] = equation[equation_name].subs("nx", f"1. / (({crop_shape[2] - 1}) * {self.dis_names[2]})")
            equation[equation_name] = sympy.lambdify(self.dim_names + self.phy_fea_names + self.dis_names, equation[equation_name], {"dif": self.torch_dif})
        self.equation = equation

    def __call__(self, point_coord, point_value, point_dis) -> torch.Tensor:
        point_coord = [point_coord[..., i: i+1] for i in range(point_coord.shape[-1])]
        point_coord[0].requires_grad = True
        point_coord[1].requires_grad = True
        point_coord[2].requires_grad = True
        point_value = [point_value[..., i: i+1] for i in range(point_value.shape[-1])]
        point_dis = [point_dis[..., i: i+1] for i in range(point_dis.shape[-1])]
        phy_fea_dim_dis = point_coord + point_value + point_dis
        values = []
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = phy_fea_dim_dis 
        for equation_name in self.equation.keys():
            value = self.equation[equation_name](x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)
            values.append(value)
        values = torch.stack(values, dim=0)
        return values

def main():
    from nptr import NptrFactory
    from sampler import Sampler
    from croper import Croper
    from eage_point_generator_factory import EagePointGeneratorFactory

    use_nptr = True
    use_eage_sampling = True
    nptr = NptrFactory()(use_nptr, "nptr")
    point_generator_factory = EagePointGeneratorFactory()
    point_generator = point_generator_factory(use_eage_sampling, "eage_sample_v1", 7.0, 0.5)
    sampler = Sampler([16, 128, 128], [4, 8, 8], 512, point_generator)
    croper = Croper([256, 128, 512], [16, 128, 128],)
    dataset = OceanDataset("data", ["isw_dataset_c307.npz"], ["p", "b", "u", "w"], nptr, sampler, croper)
    model = PdeModel(["p", "b", "u", "w"], os.path.join("equation", "navier_stokes.json"), dataset)

if __name__ == "__main__":
    main()
