import logging
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


class Logger:
    _instance = None

    def __new__(cls, cfg_logger: dict):
        if cls._instance is None:
            cls._instance = super(Logger, cls).__new__(cls)

            log_dir = Path(cfg_logger.log_dir)
            log_dir.mkdir(exist_ok=True)

            # Python logger
            cls._instance.logger = logging.getLogger(cfg_logger.logger_name)
            cls._instance.logger.setLevel(cfg_logger.logger_level)
            cls._instance.logger.handlers.clear()

            file_handler = logging.FileHandler(log_dir / cfg_logger.logger_filename)
            console_handler = logging.StreamHandler()

            formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            cls._instance.logger.addHandler(file_handler)
            cls._instance.logger.addHandler(console_handler)

            # TensorBoard SummaryWriter
            cls._instance.writer = SummaryWriter(log_dir=cfg_logger.tensorboard_dir)

        return cls._instance

    @classmethod
    def get_logger(cls):
        return cls._instance.logger

    @classmethod
    def get_writer(cls):
        return cls._instance.writer
