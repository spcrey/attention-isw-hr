from abc import ABC, abstractmethod
import json
import os

import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

class MetricStrategy(ABC):
    @abstractmethod
    def calc(self, ground_truth: np.ndarray, prediction: np.ndarray):
        pass

class PsnrStrategy(MetricStrategy):
    def calc(self, ground_truth: np.ndarray, prediction: np.ndarray):
        return peak_signal_noise_ratio(ground_truth, prediction, data_range=1)
    
class SsimStrategy(MetricStrategy):
    def calc(self, ground_truth: np.ndarray, prediction: np.ndarray):
        return structural_similarity(ground_truth, prediction, data_range=1)

class NormStrategy(ABC):
    @abstractmethod
    def calc(self, ground_truth: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray]:
        pass

class UseNorm(NormStrategy):
    def calc(self, ground_truth: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray]:
        vmin = np.min(ground_truth)
        vmax = np.max(ground_truth)
        new_ground_truth = (ground_truth - vmin) / (vmax - vmin)
        new_prediction = (prediction - vmin) / (vmax - vmin)
        return new_ground_truth, new_prediction
    
class NoNorm(NormStrategy):
    def calc(self, ground_truth: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray]:
        return ground_truth, prediction

class BatchFeaMetricCalculator:
    def set_metric_strategy(self, strategy: MetricStrategy):
        self.metric_strategy = strategy

    def set_norm_strategy(self, strategy: NormStrategy):
        self.norm_strategy = strategy

    def calc(self, bf_ground_truth: np.ndarray, bf_prediction: np.ndarray) -> list[float]:
        # shape = [b*, c, t, z, x], b* = (b1, b2, ...)
        shape = bf_ground_truth.shape
        # [(b1, b2, ...), c, t, z, x] -> [b, c, t, z, x], b = b1 * b2 * ...
        bf_ground_truth = bf_ground_truth.reshape(-1, *shape[-4:])
        bf_prediction = bf_prediction.reshape(-1, *shape[-4:])
        shape = bf_ground_truth.shape
        batch_size, fea_num = shape[:2]
        fea_metrics = []
        for fea_id in range(fea_num):
            # [b, c, t, z, x] -> [b, t, z, x]
            b_ground_truth = bf_ground_truth[:,fea_id]
            b_prediction = bf_prediction[:,fea_id]
            batch_metrics = []
            for item_id in range(batch_size):
                # [b, t, z, x] -> [t, z, x]
                ground_truth = b_ground_truth[item_id]
                prediction = b_prediction[item_id]
                ground_truth, prediction = self.norm_strategy.calc(ground_truth, prediction)
                metric = self.metric_strategy.calc(ground_truth, prediction)
                batch_metrics.append(metric)
            fea_metrics.append(np.mean(batch_metrics))
        return fea_metrics

def test_metric_calc():
    shape = [4, 11,13,15]
    ground_truth = np.random.rand(*shape)
    prediction = np.random.rand(*shape)
    calculator = BatchFeaMetricCalculator()
    calculator.set_metric_strategy(PsnrStrategy())
    calculator.set_norm_strategy(UseNorm())
    fea_psnr = calculator.calc(ground_truth, prediction)
    calculator.set_metric_strategy(SsimStrategy())
    fea_ssim = calculator.calc(ground_truth, prediction)
    pass

class Metric(ABC):
    def __init__(self, phy_fea_names, file_folder):
        self.phy_fea_names = phy_fea_names
        self.file_folder = file_folder
        os.makedirs(file_folder, exist_ok=True)

    @abstractmethod
    def add_psnr_ssim(self, psnr: list[float], ssim: list[float]):
        pass

    @abstractmethod
    def save(self):
        pass

# TrainMetric
# └── data
#     ├── reg_loss
#     │   └── epoch_sequence (list)
#     │       ├── 0
#     │       │   ├── batch_sequence (list)
#     │       │   └── average
#     │       └── (...)
#     ├── pde_loss (...)
#     ├── sum_loss (...)
#     ├── psnr
#     │   └── epoch_sequence (list)
#     │       ├── 0
#     │       │   ├── p
#     │       │   │   ├── batch_sequence (list)
#     │       │   │   └── average
#     │       │   ├── b (...)
#     │       │   ├── u (...)
#     │       │   ├── w (...)
#     │       │   ├── average
#     │       │   └── min
#     │       └── (...)
#     └── ssim (...)

class TrainMetric(Metric):
    def __init__(self, phy_fea_names: list[str], file_folder: str) -> None:
        super().__init__(phy_fea_names, file_folder)
        self.loss_names = ["reg_loss", "pde_loss", "sum_loss"]
        self.data = {
            metric_name: {"epoch_sequence": []} for metric_name in [*self.loss_names, "psnr", "ssim"]
        }

    def new_epoch(self) -> None:
        for metric_name in self.loss_names:
            self.data[metric_name]["epoch_sequence"].append(
                {
                    "batch_sequence": [],
                    "average": None
                }
            )
        for metric_name in ["psnr", "ssim"]:
            self.data[metric_name]["epoch_sequence"].append({
                ** {
                    phy_fea_name: {
                        "batch_sequence": [],
                        "average": None
                    } for phy_fea_name in self.phy_fea_names
                },
                "average": None,
                "min": None,
            })

    def add_loss(self, reg_loss: float, pde_loss: float, sum_loss: float) -> None:
        for metric_name, loss in zip(self.loss_names, [reg_loss, pde_loss, sum_loss]):
            self.data[metric_name]["epoch_sequence"][-1]["batch_sequence"].append(loss)

    def add_psnr_ssim(self, fea_psnr: list[float], fea_ssim: list[float]) -> None:
        for phy_fea_index, phy_fea_name in enumerate(self.phy_fea_names):
            self.data["psnr"]["epoch_sequence"][-1][phy_fea_name]["batch_sequence"].append(fea_psnr[phy_fea_index])
            self.data["ssim"]["epoch_sequence"][-1][phy_fea_name]["batch_sequence"].append(fea_ssim[phy_fea_index])

    def average(self, data: list[float]) -> float:
        return sum(data) / len(data)

    def update(self):
        for metric_name in self.loss_names:
            batch_sequence = self.data[metric_name]["epoch_sequence"][-1]["batch_sequence"]
            self.data[metric_name]["epoch_sequence"][-1]["average"] = self.average(batch_sequence)
        for metric_name in ["psnr", "ssim"]:
            phy_fea_averages = []
            for phy_fea_name in self.phy_fea_names:
                batch_sequence = self.data[metric_name]["epoch_sequence"][-1][phy_fea_name]["batch_sequence"]
                average = self.average(batch_sequence)
                self.data[metric_name]["epoch_sequence"][-1][phy_fea_name]["average"] = average
                phy_fea_averages.append(average)
            self.data[metric_name]["epoch_sequence"][-1]["average"] = self.average(phy_fea_averages)
            self.data[metric_name]["epoch_sequence"][-1]["min"] = min(phy_fea_averages)

    def save(self):
        with open(os.path.join(self.file_folder, "metric.json"), "w") as file:
            json.dump(self.data, file, indent=4)

    def get_eopch_index(self):
        return len(self.data["sum_loss"]["epoch_sequence"])

    def load(self, file_folder):
        with open(os.path.join(file_folder, "metric.json"), "r") as file:
            self.data = json.load(file)

    def get_latest_average_loss(self) -> tuple[float]:
        return tuple(
            self.data[metric_name]["epoch_sequence"][-1]["average"] for metric_name in ["reg_loss", "pde_loss", "sum_loss"]
        )
    
    def get_latest_average_psnr(self) -> float:
        return self.data["psnr"]["epoch_sequence"][-1]["average"]
    
    def get_latest_min_psnr(self) -> tuple[float]:
        return self.data["psnr"]["epoch_sequence"][-1]["min"]
    
    def get_latest_average_ssim(self) -> tuple[float]:
        return self.data["ssim"]["epoch_sequence"][-1]["average"]
    
    def get_latest_min_ssim(self) -> tuple[float]:
        return self.data["ssim"]["epoch_sequence"][-1]["min"]
        
    def is_best_metric(self, metric_name: str, type="average"):
        if len(self.data[metric_name]["epoch_sequence"])==0:
            raise "not data"
        if len(self.data[metric_name]["epoch_sequence"])==1:
            return True
        epoch_metrics = [
            self.data[metric_name]["epoch_sequence"][epoch_index][type] for epoch_index in range(len(self.data[metric_name]["epoch_sequence"]))
        ]
        fun = min if metric_name in self.loss_names else max
        if self.data[metric_name]["epoch_sequence"][-1][type] == fun(epoch_metrics):
            return True
        else:
            return False

class EvalMetric(Metric):
    def __init__(self, phy_fea_names, file_folder) -> None:
        super().__init__(phy_fea_names, file_folder)
        self.data = {
            "psnr": {},
            "ssim": {},
        }

    def add_psnr_ssim(self, psnr: list[float], ssim: list[float]):
        for phy_fea_index, phy_fea_name in enumerate(self.phy_fea_names):
            self.data["psnr"][phy_fea_name] = psnr[phy_fea_index]
            self.data["ssim"][phy_fea_name] = ssim[phy_fea_index]

    def save(self):
        with open(os.path.join(self.file_folder, "metric.json"), "w") as file:
            json.dump(self.data, file, indent=4)
    
def test_metric():
    train_metric = TrainMetric(["p", "b", "u", "w"], os.path.join("log", "metric_test"))
    # train_metric.load(os.path.join("log", "metric_test"))
    train_metric.new_epoch()
    train_metric.add_loss(1,3,5)
    train_metric.add_loss(2,2,4)
    train_metric.add_loss(2,2,4)
    train_metric.add_psnr_ssim([30, 20, 6, 34], [20, 24, 55, 34])
    train_metric.add_psnr_ssim([31, 21, 11, 31], [22, 23, 100, 32])
    train_metric.update()
    print("eopch_index:", train_metric.get_eopch_index())
    print("latest_average_psnr:", train_metric.get_latest_average_psnr())
    print("latest_average_ssim:", train_metric.get_latest_average_ssim())
    print("latest_average_loss:", train_metric.get_latest_average_loss())
    print("is_best_sum_loss:", train_metric.is_best_metric("sum_loss"))
    print("is_best_psnr:", train_metric.is_best_metric("psnr"))
    print("is_best_ssim:", train_metric.is_best_metric("ssim"))
    train_metric.new_epoch()
    train_metric.add_loss(1,4,3)
    train_metric.add_loss(2,2,1)
    train_metric.add_loss(2,2,4)
    train_metric.add_psnr_ssim([33, 23, 22, 36], [20, 24, 55, 34])
    train_metric.add_psnr_ssim([31, 21, 101, 31], [11, 12, 22, 32])
    train_metric.update()
    print("eopch_index:", train_metric.get_eopch_index())
    print("latest_average_psnr:", train_metric.get_latest_average_psnr())
    print("latest_average_ssim:", train_metric.get_latest_average_ssim())
    print("latest_average_loss:", train_metric.get_latest_average_loss())
    print("is_best_sum_loss:", train_metric.is_best_metric("sum_loss"))
    print("is_best_psnr:", train_metric.is_best_metric("psnr"))
    print("is_best_ssim:", train_metric.is_best_metric("ssim"))
    train_metric.save()
    eval_metric = EvalMetric(["p", "b", "u", "w"], os.path.join("log", "metric_test", "eval"))
    eval_metric.add_psnr_ssim([30, 20, 6, 34], [20, 24, 55, 34])
    eval_metric.save()

def main():
    test_metric_calc()

if __name__ == "__main__":
    main()