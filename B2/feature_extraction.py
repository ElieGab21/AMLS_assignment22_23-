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

    :Args:
        img_path: the path to the images folder
        labels_path: the path to the labels.csv file

    :return:
        landmark_features:  an array containing 68 landmark points for each image in which a face was detected
        gender_labels:      an array containing the gender label (male=0 and female=1) for each image in
                            which a face was detected

    labels is index, filename, gender, smiling
    """
    image_paths = [os.path.join(img_path, l) for l in os.listdir(img_path)]
    target_size = None
    labels_file = open(os.path.join(labels_path, LABELS_FILENAME), 'r')

    labels_df = pd.read_csv(labels_file, sep='\t')
    eye_color_labels = {row['file_name'] : row['eye_color'] for index, row in labels_df.iterrows()}

    if os.path.isdir(img_path):
        all_images = []
        all_labels = []

        img_nmb = 0

        for img_path in image_paths:

            # if img_nmb > 4:
            #     break

            file_name = img_path.split('/')[-1]

            print(file_name)

            # load image
            # img = image.img_to_array(
            #     image.load_img(img_path,
            #                    target_size=target_size,
            #                    interpolation='bicubic')).astype('uint8')

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            img = cv2.resize(img, (64, 64))

            all_images.append(img)
            all_labels.append(eye_color_labels[file_name])

            # img_nmb+=1

    images = np.array(all_images)
    eye_labels = np.array(all_labels)

    return images, eye_labels