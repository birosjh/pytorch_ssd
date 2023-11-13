from pathlib import Path

from loggers.base_logger import BaseLogger


class TextFileLogger(BaseLogger):
    """
    A class that handles logging to a text file
    """

    def __init__(self):
        self.folderpath = Path("logs")
        self.folderpath.mkdir(parents=True, exist_ok=True)

        self.filepath = self.folderpath.joinpath("log.txt")

    def log(self, records: dict, epoch: int) -> None:
        """
        Log records to text file

        Args:
            records (dict): Values to be recorded
            epoch (int): Current Epoch
        """

        if epoch == 0:
            record_names = list(records.keys())

            self.create_text_file(record_names)

        line = f"{epoch}"

        for record_name, record_value in records.items():
            line += f",{record_value}"

        line += "\n"

        with open(str(self.filepath), "a") as writer:
            writer.write(line)

    def create_text_file(self, record_names: list) -> None:
        """
        Create a text file with the necessary columns

        Args:
            record_names (list): Names of all elements to be recorded
        """

        line = "epoch," + ",".join(record_names) + "\n"

        print(line)

        with open(str(self.filepath), "w") as writer:
            writer.write(line)
