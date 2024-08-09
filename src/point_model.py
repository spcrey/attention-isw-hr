from abc import abstractmethod

from torch import nn

class PointModel(nn.Module):
    def __init__(self, n_latent_fea: int, n_phy_fea: int, n_baselayer_fea: int, n_point_model_layer: int) -> None:
        super(PointModel, self).__init__()

    @abstractmethod
    def forward(self, corner_pos, corner_value):
        pass
