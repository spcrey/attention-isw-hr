from torch import nn

class GridModel(nn.Module):
    def __init__(self, n_phy_fea: int, n_latent_fea: int, grid_shape: list[int], n_baselayer_fea: int, n_layer: int) -> None:
        super(GridModel, self).__init__()
        self.n_baselayer_fea = n_baselayer_fea
        self.n_layer = n_layer
        self.layer = nn.Conv3d(n_phy_fea, n_latent_fea, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.layer(x)
