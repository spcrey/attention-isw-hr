from typing import Optional
import numpy as np
import torch

from cuda_manager import CudaManager
from grid_model_factory import GridModel, GridModelFactory
from pde_model import PdeModel
from point_coord_iterator import PointCoordIterator
from point_model import PointModel
from point_model_factory import PointModelFactory

class PointGridInterpolator:
    def split(self, grid_data: torch.Tensor, point_coord: torch.Tensor) -> torch.Tensor:
        xmin=0.
        xmax=1.
        def clip_tensor(input_tensor, xmin, xmax):
            """Clip tensor per by per column bounds."""
            return torch.max(torch.min(input_tensor, xmax), xmin)
        grid = grid_data.permute(0, 2, 3, 4, 1)
        query_pts = point_coord
        device = grid.device
        dim = len(grid.shape) - 2
        size = torch.tensor(grid.shape[1:-1]).float().to(device)

        # convert xmin and xmax
        if isinstance(xmin, (int, float)) or isinstance(xmax, (int, float)):
            xmin = float(xmin) * torch.ones([dim], dtype=torch.float32, device=grid.device)
            xmax = float(xmax) * torch.ones([dim], dtype=torch.float32, device=grid.device)
        elif isinstance(xmin, (list, tuple, np.ndarray)) or isinstance(xmax, (list, tuple, np.ndarray)):
            xmin = torch.tensor(xmin).to(grid.device)
            xmax = torch.tensor(xmax).to(grid.device)

        # clip query_pts
        eps = 1e-6 * (xmax - xmin)
        query_pts = clip_tensor(query_pts, xmin+eps, xmax-eps)

        cubesize = (xmax - xmin) / (size - 1)
        ind0 = torch.floor(query_pts / cubesize).long()  # (batch, num_points, dim)
        ind1 = ind0 + 1
        ind01 = torch.stack((ind0, ind1), dim=0) # (2, batch, num_points, dim)
        tmp = torch.tensor([0, 1], dtype=torch.long)
        com_ = torch.stack(torch.meshgrid(tuple([tmp] * dim)), dim=-1).view(-1, dim)
        dim_ = torch.arange(dim).repeat(com_.shape[0], 1) # (2**dim, dim)
        ind_ = ind01[com_, ..., dim_]   # (2**dim, dim, batch, num_points)
        ind_n = ind_.permute(2, 3, 0, 1) # (batch, num_points, 2**dim, dim)
        ind_b = (torch.arange(grid.shape[0])
                .expand(ind_n.shape[1], ind_n.shape[2], grid.shape[0])
                .permute(2, 0, 1))  # (batch, num_points, 2**dim)

        # latent code on neighbor nodes
        unpack_ind_n = tuple([ind_b] + [ind_n[..., i] for i in range(ind_n.shape[-1])])
        corner_values = grid[unpack_ind_n] # (batch, num_points, 2**dim, in_features)

        # weights of neighboring nodes
        xyz0 = ind0.float() * cubesize        # (batch, num_points, dim)
        xyz1 = (ind0.float() + 1) * cubesize  # (batch, num_points, dim)
        xyz01 = torch.stack((xyz0, xyz1), dim=0) # (2, batch, num_points, dim)
        pos = xyz01[com_, ..., dim_].permute(2, 3, 0, 1)   # (batch, num_points, 2**dim, dim)
        pos_ = xyz01[1-com_, ..., dim_].permute(2, 3, 0, 1)   # (batch, num_points, 2**dim, dim)
        dxyz_ = torch.abs(query_pts.unsqueeze(-2) - pos_) / cubesize # (batch, num_points, 2**dim, dim)
        weights = torch.prod(dxyz_, dim=-1, keepdim=False) # (batch, num_points, 2**dim)
        x_relative = (query_pts.unsqueeze(-2) - pos) / cubesize # (batch, num_points, 2**dim, dim)

        return x_relative, corner_values, weights, 

    def merge(self, corner_value, corner_weight):
        point_value = torch.sum(corner_value * corner_weight.unsqueeze(-1), axis=-2)
        return point_value

class TripleModel:
    def __init__(self, grid_model: GridModel, point_model: PointModel, pde_model: PdeModel) -> None:
        self.cuda_manager = CudaManager()
        self.grid_model = self.cuda_manager.model_cuda_adapt(grid_model)
        self.point_model = self.cuda_manager.model_cuda_adapt(point_model)
        self.pde_model = pde_model
        self.point_grid_interpolator = PointGridInterpolator()
        self.calc_pde = False
        self.point_iterate = False
        self.point_batch_size = None

    def set_calc_pde(self, calc_pde: bool) -> None:
        self.calc_pde = calc_pde

    def set_point_iterate(self, point_iterate: bool, point_batch_size: Optional[int|None] = None) -> None:
        self.point_iterate = point_iterate
        self.point_batch_size = point_batch_size

    def __call__(self, phy_fea_grid: torch.Tensor, point_coord: torch.Tensor, point_sod=None, point_dis=None):
        # parallel
        grid_model = self.cuda_manager.model_parallel_adapt(self.grid_model)
        point_model = self.cuda_manager.model_parallel_adapt(self.point_model)
        # forward
        latent_grid = grid_model(phy_fea_grid)
        if self.point_iterate:
            point_coord_iterator = PointCoordIterator(point_coord, self.point_batch_size)
            phy_fea_point_values = []
            while point_coord_iterator.has_next():
                batch_point_coord = point_coord_iterator.next()
                # process start
                corner_pos, latent_corner_value, corner_weight = self.point_grid_interpolator.split(latent_grid, batch_point_coord)
                phy_fea_corner_value = point_model(corner_pos, latent_corner_value)
                phy_fea_point_value = self.point_grid_interpolator.merge(phy_fea_corner_value, corner_weight)
                # process end
                phy_fea_point_values.append(phy_fea_point_value)
            phy_fea_point_value = torch.concatenate(phy_fea_point_values, dim=-2)
            return phy_fea_point_value

        corner_pos, latent_corner_value, corner_weight = self.point_grid_interpolator.split(latent_grid, point_coord)
        phy_fea_corner_value = point_model(corner_pos, latent_corner_value)
        phy_fea_point_value = self.point_grid_interpolator.merge(phy_fea_corner_value, corner_weight)
        if self.calc_pde:
            pde_value = point_sod * self.pde_model(point_coord, phy_fea_point_value, point_dis)
            return phy_fea_point_value, pde_value
        else:
            return phy_fea_point_value
    
    def state_dict(self):
        return {
            "grid_model": self.grid_model.module.state_dict(),
            "point_model": self.point_model.module.state_dict(),
        }
    
    def load_state_dict(self, state_dict):
        self.grid_model.load_state_dict(state_dict["grid_model"])
        self.point_model.load_state_dict(state_dict["point_model"])

    def parameters(self):
        return list(self.grid_model.parameters()) + list(self.point_model.parameters())
    
    def train(self):
        self.grid_model.train()
        self.point_model.train()

    def eval(self):
        self.grid_model.eval()
        self.point_model.eval()

def main():
    from isw_dataset import get_test_dataset
    CudaManager(True, [0], 4)
    factory = GridModelFactory()
    grid_model = factory("fft_attention", 4, 32, (4, 16, 16), None, None)
    factory = PointModelFactory()
    point_model = factory("cube_attention", 32, 4, None, None)
    pde_model = PdeModel(["p", "b", "u", "w"], "equation/convection.json", get_test_dataset())
    triple_model = TripleModel(grid_model, point_model, pde_model)

if __name__ == "__main__":
    main()
