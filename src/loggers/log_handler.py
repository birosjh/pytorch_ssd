from src.loggers.print_logger import PrintLogger
from src.loggers.textfile_logger import TextFileLogger


class LogHandler:
    """
    A logger class that contains all logic for creating logs
    By default all values are logged to console
    """

    def __init__(self, logger_list: dict) -> None:
        self.loggers: list = [PrintLogger()]

        if logger_list["textfile"]:
            self.loggers.append(TextFileLogger())

    def __call__(self, records: dict, epoch: int) -> None:
        """
        Log records to all initialized loggers

        Args:
            records (dict): Values to be recorded
            epoch (int): Current Epoch
        """

        for logger in self.loggers:
            logger.log(records, epoch)
