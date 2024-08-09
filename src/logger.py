import logging
import os

class Logger:
    def __init__(self, name: str, file_folder=None):
        self.logger = logging.getLogger(name=name)
        self.logger.setLevel(logging.INFO)        
        if file_folder:
            os.makedirs(file_folder, exist_ok=True)
            file_path = os.path.join(file_folder, f"{name}.log")
            self._add_file_handler(file_path)
        self._add_console_handler()
        self.name = name
        self.file_folder = file_folder
        self.content_interval = "    "
        
    def _add_file_handler(self, file_path):
        handler = logging.FileHandler(file_path)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s] [%(name)s %(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def _add_console_handler(self):
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        console.setFormatter(formatter)
        self.logger.addHandler(console)

    def info(self, message):
        self.logger.info(message)

    def info_eopch_metric(self, epoch_index, reg_loss, pde_loss, sum_loss, psnr, ssim):
        content = f"epoch_index={epoch_index}:"
        content += self.content_interval + "reg_loss={:.6f}".format(reg_loss)
        content += self.content_interval + "pde_loss={:.6f}".format(pde_loss)
        content += self.content_interval + "sum_loss={:.6f}".format(sum_loss)
        content += self.content_interval + "psnr={:.6f}".format(psnr)
        content += self.content_interval + "ssim={:.6f}".format(ssim)
        self.info(content)

def main():
    logger = Logger("test", file_folder=os.path.join("log", "logger_test"))
    logger.info("Hello World!")

if __name__ == "__main__":
    main()
