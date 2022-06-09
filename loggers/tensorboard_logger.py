from torch.utils.tensorboard import SummaryWriter

from loggers.base_logger import BaseLogger


class TensorboardLogger(BaseLogger):
    """
    A class that handles recording logs to tensorboard
    """

    def __init__(self) -> None:

        self.writer = SummaryWriter()

    def log(self, records: dict, epoch: int) -> None:
        """
        Log records to tensorboard

        Args:
            records (dict): Values to be recorded
            epoch (int): Current Epoch
        """

        for record_name, record_value in records.items():

            self.writer.add_scalar(record_name, record_value, epoch)
