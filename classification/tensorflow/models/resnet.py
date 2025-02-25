""" Resnet implementation model
"""
from .model import BaseModel
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Add

class ResnetFromScratch(BaseModel):
    