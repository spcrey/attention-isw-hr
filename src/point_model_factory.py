import sys

from point_model import PointModel

sys.path.append("approach")

class PointModelFactory:
    def __call__(self, name: str, n_latent_fea: int, n_phy_fea: int, n_baselayer_fea: int, n_point_model_layer: int) -> PointModel:
        if name == "cube_attention":
            from cube_attention import CubeAttention
            return  CubeAttention(n_latent_fea, n_phy_fea, n_baselayer_fea, n_point_model_layer)
        else:
            raise f"no model named {type}"

def main():
    import torch
    model_factory = PointModelFactory()
    model = model_factory("cube_attention", 32, 4, 16, 4)
    x = torch.rand(10, 8, 32)
    y = model(x)
    print("output shape:", y.shape)

if __name__ == "__main__":
    main()
