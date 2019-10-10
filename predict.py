from fastai.vision import *
import torch
from fastai.metrics import error_rate
bs = 64
path = "/Users/Sai/Desktop/ML/Model"
learn = load_learner(path, 'resnet50_model.pkl')
mia1 = open_image("/Users/Sai/Desktop/ML/Model/cat3.jpg")
pred = learn.predict(mia1);
print (pred[0])
