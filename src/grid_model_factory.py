import sys

from grid_model import GridModel

sys.path.append("approach")

class GridModelFactory:
    def __call__(self, name: str, n_phy_fea: int, n_latent_fea: int, grid_shape: list[int], n_baselayer_fea: int, n_layer: int) -> GridModel:
        if name == "fft_attention":
            from fft_attention import FftAttention
            return  FftAttention(n_phy_fea, n_latent_fea, grid_shape, n_baselayer_fea, n_layer)
        else:
            raise f"no model named {type}"

def main():
    import torch
    model_factory = GridModelFactory()
    model = model_factory("fft_attention", 4, 32, (4, 16, 16), 16, 4)
    x = torch.rand(10, 4, 4, 16, 16)
    y = model(x)
    print("output shape:", y.shape)

if __name__ == "__main__":
    main()
