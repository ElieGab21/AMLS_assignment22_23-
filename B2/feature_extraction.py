import os
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import dlib

LABELS_FILENAME = 'labels.csv'

detector = dlib.get_frontal_face_detector()
predictor_path = os.path.join(os.sys.path[0], 'B2/shape_predictor_68_face_landmarks.dat')
predictor = dlib.shape_predictor(predictor_path)

def run_dlib_extraction(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

    detect = detector(gray ,1)
    num_faces = len(detect)

    if num_faces == 0:
        return None

    shape = predictor(gray, detect[0])

    # Retrieving right eye from the image
    x1=shape.part(42).x 
    x2=shape.part(45).x 
    y1=shape.part(43).y 
    y2=shape.part(46).y

    righteye = img[y1:y2, x1:x2]

    return righteye


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

            if img_nmb > 0:
                break

            file_name = img_path.split('/')[-1]

            print(file_name)

            img = cv2.imread(img_path)

            features = run_dlib_extraction(img)

            if features is not None:
                all_images.append(features)
                all_labels.append(eye_color_labels[file_name])

            img_nmb+=1

    images = np.array(all_images)
    eye_labels = np.array(all_labels)

    return images, eye_labels