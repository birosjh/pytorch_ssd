import torch
import yaml

from datasets.image_dataset import ImageDataset
from models.ssd import SSD
from trainer.trainer import Trainer
from utils.data_encoder import DataEncoder
from visualize.visualize_batch import visualize_bbox


def inference(config_path: str, image_path: str, output_path: str) -> None:
    """
    A function to run inference with the SSD model

    Args:
        config_path (str): Path to desired config file
    """

    with open(config_path) as file:
        config = yaml.safe_load(file)

    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    model_config = config["model_configuration"]
    inference_config = config["inference_configuration"]

    num_classes = len(data_config["classes"]) + 1

    model = SSD(model_config, num_classes).to(device)
    model.load_state_dict(
        torch.load(inference_config["model_path"])
    )
    model.eval()

    transform = A.Resize(
        inference_config["figure_size"],
        inference_config["figure_size"]
    )

    image = cv2.imread(image_path)
    image = transform(transform)["image"]

    predictions = model(image)

    np_image = image.permute(1, 2, 0).numpy().copy().astype(np.uint8)

    np_labels = predictions[labelset[:, -1] > 0].numpy().astype(int)

    for label in np_labels:

        np_image = visualize_bbox(
            np_image, label, color_list, data_config["classes"]
        )

    cv2.imwrite(output_path, batch_image)


