import json
import torch
import torch.utils.data as data

class DataLoader(data.dataset):

    def __init__(self, data_config, batch_size):

        self.batch_size = batch_size

        with open(data_config.annotations) as file:
            annotations_file = file.read()

        raw_annotations = json.loads(annotations_file)

        self.images_with_ids = self.prep_images(
            raw_annotations["images"]
        )

        self.image_ids_with_annotations = self.prep_images(
            raw_annotations["annotations"]
        )

        self.categories_by_id = self.prep_categories(
            raw_annotations["categories"]
        )


    def prep_images(self, images_json):

        images_with_ids = {}

        for image in images_json:
            images_with_ids[image["file_name"]] = image["id"]

        return images_with_ids

    def prep_annotations(self, annotations_json):

        image_ids_with_annotations = {}

        for annotation in annotations_json:
            key = annotation['image_id']
            value = annotation['bbox'] + [annotation['category_id']]

            if key in image_ids_with_annotations.keys():

                image_ids_with_annotations[key].append(value)

            else:
                image_ids_with_annotations[key] = [value]

        return image_ids_with_annotations

    def prep_categories(self, categories_json):

        categories_by_id = {}

        for category in categories_json:
            categories_by_id[category['id']] = category['name']

        return categories_by_id
