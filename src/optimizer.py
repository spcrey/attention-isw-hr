from torch import optim
from torch.optim import lr_scheduler

from triple_model import TripleModel

class OptimizerFactory:
    def __call__(self, type: str, model: TripleModel, learning_rate: float) -> optim:
        params = model.parameters()
        if type == "sgd":
            return optim.SGD(params, lr=learning_rate)
        elif type == "adam":
            return optim.Adam(params, lr=learning_rate)
        else:
            raise f"no optimizer type named {type}"
        
class Scheduler(lr_scheduler.ReduceLROnPlateau):
    def __init__(self, optimizer: optim.Optimizer) -> None:
        super().__init__(optimizer, "min")
