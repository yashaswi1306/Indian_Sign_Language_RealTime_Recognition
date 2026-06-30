import torch
from torchvision import transforms
import json

from PIL import Image

from model_isl import ISL_CNN
with open('class_names.json','r') as f:
  class_names=json.load(f)

model_0=ISL_CNN(len(class_names))
model_0.load_state_dict(torch.load('/content/drive/MyDrive/isl_model_2.pth',map_location="cpu"))
model_0.eval()

transform_img=transforms.Compose(
   [
    transforms.Resize((128,128)),
    transforms.ToTensor(), # pixels lie between 0 and 1
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]) # pixels lie btw -1 -> 1. [0.5,0.5,0.5] for [R,G,B]
    ]
    )

# FASTAPI part

# fron fastapi import FastAPI

# app=FastAPI()
