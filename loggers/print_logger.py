from loggers.base_logger import BaseLogger


class PrintLogger(BaseLogger):
    """
    A class that handles printing logs to the console
    """

    def __init__(self) -> None:
        pass

    def log(self, records: dict, epoch: int) -> None:
        """
        Log records to console

        Args:
            records (dict): Values to be recorded
            epoch (int): Current Epoch
        """

        print(f"Epoch: {epoch}")

        for record_name, record_value in records.items():
            print(f"{record_name}: {record_value}")
