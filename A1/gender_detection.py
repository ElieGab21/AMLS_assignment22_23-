import A1.lab2_landmarks as l2
import numpy as np
from sklearn.metrics import classification_report,accuracy_score
from sklearn import svm
import os

def get_data():

    basedir = 'Datasets/celeba'
    img_path = os.path.join(basedir, 'img')

    X, y = l2.extract_features_labels(img_path, basedir)
    Y = np.array([y, -(y - 1)]).T
    tr_X = X[:100]
    tr_Y = Y[:100]
    te_X = X[100:]
    te_Y = Y[100:]

    return tr_X, tr_Y, te_X, te_Y

def img_SVM(training_images, training_labels, test_images, test_labels):
    classifier = 0
    classifier.fit(training_images, training_labels)
    pred = classifier.predict(test_images)
    print("Accuracy:", accuracy_score(test_labels, pred))

   # print(pred)
    return pred

def test_gender_detection():
    tr_X, tr_Y, te_X, te_Y= get_data()
    pred=img_SVM(tr_X.reshape((100, 68*2)), list(zip(*tr_Y))[0], te_X.reshape((35, 68*2)), list(zip(*te_Y))[0])