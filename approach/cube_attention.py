import sys

import torch
from torch import nn

sys.path.append("src")

from point_model import PointModel

class CubeAttention(PointModel):
    def __init__(self, n_latent_fea: int, n_phy_fea: int, n_baselayer_fea: int, n_point_model_layer: int):
        super(CubeAttention, self).__init__(n_latent_fea, n_phy_fea, n_baselayer_fea, n_point_model_layer)
        self.layer = nn.Linear(n_latent_fea + 3, n_phy_fea)

    def forward(self, corner_pos, corner_value):
        corner_value = torch.concat([corner_pos, corner_value], dim=-1)
        return self.layer(corner_value)

def main():
    import torch
    model = CubeAttention(32, 4, 16, 4)
    x = torch.rand(10, 8, 32)
    y = model(x)
    print("output shape:", y.shape)

if __name__ == "__main__":
    main()
 