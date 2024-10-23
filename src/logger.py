import logging
import os
from abc import ABC, abstractmethod

class Informer(ABC):
    @abstractmethod
    def info(self, message: str):
        pass

class StreamFileLogger(Informer):
    def __init__(self, logger: logging.Logger, file_handle: logging.FileHandler, stream_handler: logging.StreamHandler):
        logger.addHandler(file_handle)
        logger.addHandler(stream_handler)
        self._logger = logger

    def info(self, message: str):
        self._logger.info(message)

    class Builder:
        def __init__(self, name: str="log", file_folder: str="."):
            self._name = name
            self._file_folder = file_folder
            self._logger = None
            self._file_handler = None
            self._stream_handler = None

        def set_file_handler(self, file_handler: logging.FileHandler):
            self._file_handler = file_handler
            return self

        def set_stream_handler(self, stream_handler: logging.StreamHandler):
            self._stream_handler = stream_handler
            return self

        def _create_logger(self):
            if self._logger == None:
                logger_builder = NameLoggerBuilder(self._name)
                self._logger = logger_builder.build()
        
        def _create_stream_handler(self):
            if self._stream_handler == None:
                stream_handler_builder = DefaltStreamHandlerBuilder()
                self._stream_handler = stream_handler_builder.build()
                    
        def _create_file_handler(self):
            if self._file_handler == None:
                os.makedirs(self._file_folder, exist_ok=True)
                file_path = os.path.join(self._file_folder, f"{self._name}.log")
                file_handler_builder = FilePathFileHandlerBuilder(file_path)
                self._file_handler = file_handler_builder.build()

        def build(self):
            self._create_logger()
            self._create_stream_handler()
            self._create_file_handler()
            return StreamFileLogger(self._logger, self._file_handler, self._stream_handler)

class LoggerBuilder:
    @abstractmethod
    def build(self) -> logging.Logger:
        pass

class NameLoggerBuilder(LoggerBuilder):
    def __init__(self, name: str):
        self._name = name

    def build(self) -> logging.Logger:
        logger = logging.getLogger(self._name)
        logger.setLevel(logging.INFO)       
        return logger  

class FileHandlerBuilder(ABC):
    @abstractmethod
    def build(self) -> logging.FileHandler:
        pass

class StreamHandlerBuilder(ABC):
    @abstractmethod
    def build(self) -> logging.StreamHandler:
        pass

class FilePathFileHandlerBuilder(FileHandlerBuilder):
    def __init__(self, file_path: str):
        self._file_path = file_path

    def build(self) -> logging.FileHandler:
        handler = logging.FileHandler(self._file_path)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(asctime)s] [%(name)s %(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        return handler

class DefaltStreamHandlerBuilder(StreamHandlerBuilder):
    def build(self) -> logging.StreamHandler:
        handler = logging.StreamHandler()
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        return handler

class Logger:
    def __init__(self, name: str, file_folder="."):
        self.logger = StreamFileLogger.Builder(name, os.path.join(file_folder, name)).build()
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
    logger = Logger("text", "log")
    logger.info("Hello World!")

if __name__ == "__main__":
    main()
