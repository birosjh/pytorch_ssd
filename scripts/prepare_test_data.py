import os
from shutil import copyfile
from pathlib import Path

ANNOTATIONS: str = "Annotations"
JPEGIMAGES: str = "JPEGImages"

# Get Project Root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)).split("/")[0]

# Set Paths for Data and Test Data
data_path = Path(ROOT_DIR).joinpath("data")
test_data_path = Path(ROOT_DIR).joinpath("tests/test_data")

# Create test data paths
test_data_path.joinpath(ANNOTATIONS).mkdir(parents=True, exist_ok=True)
test_data_path.joinpath(JPEGIMAGES).mkdir(parents=True, exist_ok=True)

# Get test file names
test_data_file = test_data_path.joinpath("test_data.txt")

with open(test_data_file, "r") as outfile:

    lines = outfile.readlines()


# Copy files from data to test data
for line in lines:

    filename = line.split(" ")[0]

    copyfile(
        data_path.joinpath(ANNOTATIONS).joinpath(filename + ".xml"),
        test_data_path.joinpath(ANNOTATIONS).joinpath(filename + ".xml")
    )

    copyfile(
        data_path.joinpath(JPEGIMAGES).joinpath(filename + ".jpg"),
        test_data_path.joinpath(JPEGIMAGES).joinpath(filename + ".jpg")
    )
