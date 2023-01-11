import os
from B1.features_extraction import extract_features_labels
from sklearn.metrics import classification_report,accuracy_score
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np

def get_data():
    '''
    This function retrieves the data using the feature extraction function

    returns:
        tr_X, tr_Y, te_X, te_Y: The datasets 
    
    '''
    basedir = 'Datasets/cartoon_set'
    base_test_dir = 'Datasets/cartoon_set_test'
    img_path = os.path.join(basedir, 'img')
    img_test_path = os.path.join(base_test_dir, 'img')

    tr_X, tr_Y = extract_features_labels(img_path, basedir)
    te_X, te_Y = extract_features_labels(img_test_path, base_test_dir)

    return tr_X, tr_Y, te_X, te_Y

def predict(tr_X, tr_Y, te_X, te_Y):
    """
    This function perfoms PCA on every image and uses the new data to train the model
    Two models are tested: SoftMax regression and Support Vector Machine

    Prints the accuracy and classification for each model
    """

    # Testing SVM
    clf = Pipeline([('pca', decomposition.PCA(n_components=150, whiten=True)),
                    ('svm', svm.SVC(C=5., gamma=0.001))])

    clf.fit(tr_X, tr_Y)
    pred = clf.predict(te_X)

    print("SVM Accuracy:", accuracy_score(te_Y, pred))
    print('SVM Classification report \n', classification_report(te_Y, pred, zero_division=0))

    #Testing softmax regression
    softReg = Pipeline([('pca', decomposition.PCA(n_components=150, whiten=True)),
                        ('logistic', LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs'))])

    softReg.fit(tr_X, tr_Y)
    soft_pred = softReg.predict(te_X)

    print("SoftMax Accuracy:", accuracy_score(te_Y, soft_pred))
    print('SoftMax Classification report \n', classification_report(te_Y, soft_pred, zero_division=0))

def test_face_shape_detection():
    tr_X, tr_Y, te_X, te_Y= get_data()

    tr_X = tr_X.reshape((tr_X.shape[0], 64*64))
    te_X = te_X.reshape((te_X.shape[0], 64*64))

    predict(tr_X, tr_Y, te_X, te_Y)