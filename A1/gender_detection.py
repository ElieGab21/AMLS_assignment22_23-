import A1.lab2_landmarks as l2
import numpy as np
from sklearn.metrics import classification_report,accuracy_score
from sklearn import svm
import os

def get_data():

    basedir = 'Datasets/celeba'
    base_test_dir = 'Datasets/celeba_test'
    img_path = os.path.join(basedir, 'img')
    img_test_path = os.path.join(base_test_dir, 'img')

    X, y = l2.extract_features_labels(img_path, basedir)
    Y = np.array([y, -(y - 1)]).T

    X_test, y_test = l2.extract_features_labels(img_test_path, base_test_dir)
    Y_test = np.array([y_test, -(y_test - 1)]).T

    tr_X = X
    tr_Y = Y

    te_X = X_test
    te_Y = Y_test

    return tr_X, tr_Y, te_X, te_Y

def img_SVM(training_images, training_labels, test_images, test_labels):
    classifier = svm.SVC(kernel='poly', degree=3)
    classifier.fit(training_images, training_labels)
    pred = classifier.predict(test_images)
    print("Accuracy:", accuracy_score(test_labels, pred))

   # print(pred)
    return pred

def test_gender_detection():
    tr_X, tr_Y, te_X, te_Y= get_data()
    pred=img_SVM(tr_X.reshape((tr_X.shape[0], 68*2)), list(zip(*tr_Y))[0], te_X.reshape((te_X.shape[0], 68*2)), list(zip(*te_Y))[0])