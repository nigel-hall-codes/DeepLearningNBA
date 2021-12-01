import os
import cv2
import numpy as np
from sklearn import cluster

# for loading/processing the images
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

# models
from keras.applications.vgg16 import VGG16
from keras.models import Model

# clustering and dimension reduction
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# for everything else
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle


class Trainer:
    def __init__(self):
        self.found_players_dir = r'D:\PycharmProjects\DeepLearningNBA\found_players\Steph2Wise'

    def get_images(self):
        return np.array([cv2.imread(os.path.join(self.found_players_dir, image)) for image in os.listdir(self.found_players_dir)])


    def extract_features(self, file):
        model = VGG16()
        model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
        img = cv2.imread(file)
        img = cv2.resize(img, (224, 224))
        img = img.reshape(1, 224, 224, 3)

        imgx = preprocess_input(img)
        features = model.predict(imgx, use_multiprocessing=True)
        return features

    def get_labels(self):
        p = r"D:\PycharmProjects\DeepLearningNBA\found_players\Steph2Wise\vectors"

        data = {}

        for image in os.listdir(self.found_players_dir):
            print(f"getting features for {image}")
            try:
                feat = self.extract_features(image)
                data[image] = feat

            except Exception as e:
                print(f"{image} failed")
                print(e)
                pass

        filenames = np.array(list(data.keys()))
        feat = np.array(list(data.values()))
        feat = feat.reshape(-1, 4096)
        df = pd.read_csv('flower_labels.csv')
        label = df['label'].tolist()
        unique_labels = list(set(label))
        pca = PCA(n_components=100, random_state=22)
        pca.fit(feat)
        x = pca.transform(feat)

        print(f"Components before PCA: {f.shape[1]}")
        print(f"Components after PCA: {pca.n_components}")

        kmeans = KMeans(n_clusters=len(unique_labels), n_jobs=-1, random_state=22)
        kmeans.fit(x)

        groups = {}
        for file, cluster in zip(filenames, kmeans.labels_):
            if cluster not in groups.keys():
                groups[cluster] = []
                groups[cluster].append(file)
            else:
                groups[cluster].append(file)


        print(groups)

    def train(self):
        images = self.get_images()

        # X = images.reshape((-1,1))

        X = images

        x = preprocess_input(X)
        model = VGG16()
        # remove the output layer
        model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
        features = model.predict(image)


if __name__ == '__main__':
    t = Trainer()
    print(len(t.get_images()))