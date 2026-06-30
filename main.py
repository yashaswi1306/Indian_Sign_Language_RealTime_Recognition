import torch
from torchvision import transforms
import json
import io

from PIL import Image

from model_isl import ISL_CNN
with open('class_names.json','r') as f:
  class_names=json.load(f)

model_0=ISL_CNN(len(class_names))
model_0.load_state_dict(torch.load('isl_model_2.pth',map_location="cpu"))
model_0.eval()

transform_img=transforms.Compose(
   [
    transforms.Resize((128,128)),
    transforms.ToTensor(), # pixels lie between 0 and 1
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]) # pixels lie btw -1 -> 1. [0.5,0.5,0.5] for [R,G,B]
    ]
    )

# FASTAPI part

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

app=FastAPI()

app.mount("/static",StaticFiles(directory="static"),name="static")


@app.get('/')

def root():
  return FileResponse("static/index.html")

class FrameRequest(BaseModel):
  image:str # so that the webcam returns aan img in string

import base64

@app.post("/predict")

def make_preds(req:FrameRequest):
  img_bytes=req.image.split(',')[-1] # req cuz its from arequest. 
  # Now the image is in a way that theres a , and then the encoded image. 
  # So u split across all , and take the last string
  img_bytes=base64.b64decode(img_bytes)
  img_file=io.BytesIO(img_bytes)
  img=Image.open(img_file).convert("RGB")

  img_tensor=transform_img(img).unsqueeze(0) # so u have batch size too
  img_tensor=img_tensor.to("cpu")



  with torch.inference_mode():

    pred=model_0(img_tensor)
    prob=pred.softmax(dim=1)[0] # softmax across classes and then strip batch dimensions

    pred_idx=prob.argmax().item()
    
    pred_confidence=(prob[pred_idx].item())*100

    top_3=[]
    for i in prob.topk(3).indices.tolist():

      top_3.append({"label": class_names[i],"prediction confidence" : round(prob[i].item()*100,1)})

    return {
      "label": class_names[pred_idx],
      "prediction confidence" : round(pred_confidence,1),
      "top 3": top_3
    }