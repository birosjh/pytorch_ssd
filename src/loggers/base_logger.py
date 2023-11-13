from abc import ABC, abstractmethod


class BaseLogger(ABC):
    """
    A base logger class
    """

    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def log(self, records: dict, epoch: int) -> None:
        """
        Args:
            records (dict): Values to be recorded
            epoch (int): Current Epoch
        """
        pass
