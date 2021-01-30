import cv2
import json
import torch
import torch.utils.data as data

class DataLoader(data.dataset):

    def __init__(self, data_config, batch_size):

        self.batch_size = batch_size

        with open(data_config.annotations) as file:
            annotations_file = file.read()

        annotation_json = json.loads(annotations_file)

        self.image_annotations = self.prep_images_and_annotations(
            annotation_json
        )

        self.images_list = list(self.image_annotations.keys())

        self.categories_by_id = self.prep_categories(
            annotation_json["categories"]
        )

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):

        image_file = self.images_list[idx]

        image = cv2.imread(image_file)

        labels = self.image_annotations[image_file]

        return (image, labels)

    def prep_images_and_annotations(self, annotation_json):

        # Split annotation_json
        images = annotation_json["images"]
        annotations = annotation_json["annotations"]

        # List images by image ID
        images_by_ids = [] 

        for image in images:
            images_by_ids[image["id"]] = image["file_name"]

        # Create a list of image_annotations
        image_annotations = {}

        for annotation in annotations:

            image_name = images_by_ids[
                annotation['image_id']
            ]

            # Append category id to the end of the annotation
            annotation_with_category = annotation['bbox'] + [annotation['category_id']]

            if image_name in image_annotations.keys():

                image_annotations[image_name].append(
                    annotation_with_category
                )

            else:
                image_annotations[image_name] += [annotation_with_category]

        return image_annotations

    def pair_images_with_annotations(self, images, annotations):

        image_ids_with_annotations = {}

        

        return image_ids_with_annotations

    def prep_categories(self, categories_json):

        categories_by_id = {}

        for category in categories_json:
            categories_by_id[category['id']] = category['name']

        return categories_by_id
