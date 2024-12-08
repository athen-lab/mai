import torch
import numpy as np
from PIL import Image
from model import mai
from augmentation import preprocess_validation


def load_model_and_run_inference(img):
    mai.load_state_dict(torch.load("mai"))
    mai.eval()
    img_batch = np.expand_dims(img, axis=0)
    img_batch = torch.tensor(img_batch)
    prediction = mai(img_batch)
    prediction = torch.sigmoid(prediction)
    print(prediction)


def main():
    img = Image.open("test_images/dog.jpg")
    img = preprocess_validation(image=np.array(img))["image"]
    load_model_and_run_inference(img)
