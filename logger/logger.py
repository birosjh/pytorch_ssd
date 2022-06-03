from torch.utils.tensorboard import SummaryWriter


class Logger():
    """
    A logger class that contains all logic for creating logs
    """

    def __init__(self):

        self.tensorboard_writer = SummaryWriter()

    def log(self, records: dict, epoch: int) -> None:
        """
        Log records

        Args:
            records (dict): Values to be recorded
            epoch (int): Current Epoch
        """

        for record_name, record_value in records.items():

            self.tensorboard_writer.add_scalar(record_name, record_value, epoch)
