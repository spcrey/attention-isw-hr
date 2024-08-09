from torch import nn
import torch

class LossFunFactory:
    def __call__(self, type: str) -> torch.nn.modules.loss:
        if type == "l1":
            return nn.L1Loss()
        elif type == "mse" or type == "l2":
            return nn.MSELoss()
        else:
            raise f"no loss fun type named {type}"

def main():
    # data
    ground_truth = torch.tensor([1., 2., 5.])
    prediction = torch.tensor([3., 4., 6.])
    # loss fun
    loss_fun_factory = LossFunFactory()
    # L1
    loss_fun = loss_fun_factory("l1")
    loss = loss_fun(ground_truth, prediction)
    print(loss)
    # L2
    loss_fun = loss_fun_factory("mse")
    loss = loss_fun(ground_truth, prediction)
    print(loss)

if __name__ == "__main__":
    main()
