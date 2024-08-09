import sys

sys.path.append("approach")

from point_generator import PointGenerator, RandomPointGenerator

class EagePointGeneratorFactory:
    def __call__(self, is_use: bool, name: str, random_scale: float, eage_alpha: float) -> PointGenerator:
        if is_use:
            if name == "eage_sample_v1":
                from eage_point_generator_v1 import EagePointGeneratorV1
                return EagePointGeneratorV1(random_scale, eage_alpha)
            else:
                raise f"no eage_sample named {name}"
        else:
            return RandomPointGenerator()
