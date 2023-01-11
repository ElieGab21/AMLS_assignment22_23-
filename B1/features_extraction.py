import os
import tensorflow.keras.utils as image
from PIL import Image
import numpy as np
import pandas as pd
import cv2

LABELS_FILENAME = 'labels.csv'

def extract_features_labels(img_path, labels_path):
    """
    This funtion extracts the landmarks features for all images in the folder 'dataset/celeba'.
    It also extracts the gender label for each image.

    Args:
        img_path: the path to the images folder
        labels_path: the path to the labels.csv file

    return:
        images:  an array containing images of the eyes of each image that was detected
        face_labels: an array containing the face label for each image in which a face was detected

    """

    image_paths = [os.path.join(img_path, l) for l in os.listdir(img_path)]
    labels_file = open(os.path.join(labels_path, LABELS_FILENAME), 'r')

    labels_df = pd.read_csv(labels_file, sep='\t')
    face_shape_labels = {row['file_name'] : row['face_shape'] for index, row in labels_df.iterrows()}

    if os.path.isdir(img_path):
        all_images = []
        all_labels = []

        for img_path in image_paths:

            file_name = img_path.split('/')[-1]

            print(file_name)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            img = cv2.resize(img, (64, 64))

            all_images.append(img)
            all_labels.append(face_shape_labels[file_name])

    images = np.array(all_images)
    face_labels = np.array(all_labels)

    return images, face_labels