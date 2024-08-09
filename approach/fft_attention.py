import sys
import torch
import torch.nn as nn

sys.path.append("src")

from grid_model import GridModel

class FftConv3d(nn.Module):
    def __init__(self, n_in_fea: int, n_out_fea: int):
        super(FftConv3d, self).__init__()
        self.conv_real = nn.Conv3d(n_in_fea, n_out_fea, kernel_size=1, stride=1, padding=0)
        self.conv_imag = nn.Conv3d(n_in_fea, n_out_fea, kernel_size=1, stride=1, padding=0)

    #（b, c1, t, z, x）-> （b, c2, t, z, x）
    def forward(self, x):
        x = torch.fft.fft2(x)
        x_real, x_imag = x.real, x.imag
        x_real = self.conv_real(x_real)
        x_imag = self.conv_imag(x_imag)
        x = torch.complex(x_real, x_imag)
        x = torch.fft.ifft2(x).real
        return x

class PostpositionFusionBlock(nn.Module):
    def __init__(self, n_in_fea: int):
        super(PostpositionFusionBlock, self).__init__()
        self.ffc = FftConv3d(n_in_fea, n_in_fea)
        self.conv_k3 = nn.Conv3d(n_in_fea, n_in_fea , kernel_size=3, stride=1, padding=1)
        self.conv_k1 = nn.Conv3d(n_in_fea*2, 1 , kernel_size=1, stride=1, padding=0)

    # (b, c, t, z, x) -> (b, c, t, z, x)
    def forward(self, x):
        attetion = self.conv_k1(torch.concatenate([x, self.ffc(x)], axis=1))
        x += attetion * self.conv_k3(x)
        return x
    
class PrepositionFftBlock(nn.Module):
    def __init__(self, n_in_fea: int):
        super(PrepositionFftBlock, self).__init__()
        self.ffc = FftConv3d(n_in_fea, n_in_fea)
        self.conv = nn.Conv3d(n_in_fea, n_in_fea, kernel_size=3, stride=1, padding=1)

    # (b, c, t, z, x) -> (b, c, t, z, x)
    def forward(self, x):
        x = x + self.ffc(x) + self.conv(x)
        return x

class FftAttentionBlock(nn.Module):
    def __init__(self, n_in_fea: int):
        super(FftAttentionBlock, self).__init__()
        self.preposition_model = PrepositionFftBlock(n_in_fea)
        self.postpostion_model = PostpositionFusionBlock(n_in_fea)

    # (b, c, t, z, x) -> (b, c, t, z, x)
    def forward(self, x):
        x = self.preposition_model(x)
        x = self.preposition_model(x)
        return x

class FftAttention(GridModel):
    def __init__(self, n_phy_fea: int, n_latent_fea: int, grid_shape: list[int], n_baselayer_fea: int, n_layer: int):
        super(FftAttention, self).__init__(n_phy_fea, n_latent_fea, grid_shape, n_baselayer_fea, n_layer)
        self.fft_attention_block = FftAttentionBlock(n_phy_fea)
        self.conv = nn.Conv3d(n_phy_fea, n_latent_fea, 3, 1, 1)

    def forward(self, x):
        return self.conv(self.fft_attention_block(x))

def main():
    model = FftAttention(4, 32, (4, 16, 16), 16, 4)
    x = torch.rand(10, 4, 4, 16, 16)
    y = model(x)
    print("output shape:", y.shape)

if __name__ == "__main__":
    main() 
