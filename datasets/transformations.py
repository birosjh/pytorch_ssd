import albumentations as A

class Transformations():

    def __init__(self, config, mode):
            
        if config["transform"] and mode == "train":
            print("hi")
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.5),
                A.Resize(
                    config["figure_size"], config["figure_size"]
                )
            ], bbox_params=A.BboxParams(format='pascal_voc'))
        else:
            self.transform = A.Compose([
                A.Resize(
                    config["figure_size"], config["figure_size"]
                )
            ], bbox_params=A.BboxParams(format='pascal_voc'))
    
    def __call__(self, image, bounding_boxes):

        return self.transform(image=image, bboxes=bounding_boxes)
