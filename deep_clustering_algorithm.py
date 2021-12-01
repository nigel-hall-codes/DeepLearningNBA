from time import time
import keras.backend as K
from tensorflow.keras.layers import Layer, InputSpec
from keras.layers import Dense, Input
from keras.models import Model
from tensorflow.keras.optimizers import SGD
from keras import callbacks
from keras.initializers import VarianceScaling
from sklearn.cluster import KMeans
# import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import p

class ClusteringAlgorithm:
    def __init__(self, image_folder_path):
        self.image_folder_path = image_folder_path


    def found_players_image_locations(self):
        return [r"D:\PycharmProjects\DeepLearningNBA\found_players\{}".format(x) for x in os.listdir(r"D:\PycharmProjects\DeepLearningNBA\found_players")]

    def found_player_images(self):
        image_locations = self.found_players_image_locations()
        return [cv2.imread(image) for image in image_locations]


