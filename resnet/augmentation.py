import albumentations as a
import numpy as np
from albumentations.pytorch import ToTensorV2
from hyperparams import CROP_SIZE


preprocess_training = a.Compose(
    [
        a.augmentations.PadIfNeeded(min_width=CROP_SIZE, min_height=CROP_SIZE),
        a.RandomCrop(width=CROP_SIZE, height=CROP_SIZE),
        a.GaussNoise(),
        a.Flip(p=0.5),
        a.RandomRotate90(p=0.5),
        a.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)
preprocess_validation = a.Compose(
    [
        a.augmentations.PadIfNeeded(min_width=CROP_SIZE, min_height=CROP_SIZE),
        a.CenterCrop(width=CROP_SIZE, height=CROP_SIZE),
        a.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)


def transform_training(example):
    transformed = []
    for pil_image in example["image"]:
        array = np.array(pil_image.convert("RGB"))
        # check if image is in (height, width, channel) shape
        # if not, do a transpose
        if array.shape[-1] != 3:
            array = np.transpose(array, (1, 2, 0))
        img = preprocess_training(image=array)["image"]
        transformed.append(img)
    example["pixel_values"] = transformed
    return example


def transform_validation(example):
    transformed = []
    for pil_image in example["image"]:
        array = np.array(pil_image.convert("RGB"))
        if array.shape[-1] != 3:
            array = np.transpose(array, (1, 2, 0))
        img = preprocess_validation(image=array)["image"]
        transformed.append(img)
    example["pixel_values"] = transformed
    return example
