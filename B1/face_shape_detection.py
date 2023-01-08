import os
from B1.features_extraction import extract_features_labels
from sklearn.metrics import classification_report,accuracy_score
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

def get_data():

    basedir = 'Datasets/cartoon_set'
    base_test_dir = 'Datasets/cartoon_set_test'
    img_path = os.path.join(basedir, 'img')
    img_test_path = os.path.join(base_test_dir, 'img')

    tr_X, tr_Y = extract_features_labels(img_path, basedir)
    te_X, te_Y = extract_features_labels(img_test_path, base_test_dir)

    return tr_X, tr_Y, te_X, te_Y

def test_face_shape_detection():
    tr_X, tr_Y, te_X, te_Y= get_data()

    tr_X = tr_X.reshape((tr_X.shape[0], 64*64))
    te_X = te_X.reshape((te_X.shape[0], 64*64))

    print('Starting PCA')
    pca = decomposition.PCA(n_components=150, whiten=True)
    pca.fit(tr_X)

    print('Transforming data')
    X_train_pca = pca.transform(tr_X)
    X_test_pca = pca.transform(te_X)

    print('Fitting model')
    # clf = svm.SVC(C=5., gamma=0.001)
    # clf.fit(X_train_pca, tr_Y)

    softReg = LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs')
    softReg.fit(X_train_pca, tr_Y)

    print('Predicting')
    # pred = clf.predict(X_test_pca)
    pred = softReg.predict(X_test_pca)

    print("Accuracy:", accuracy_score(te_Y, pred))
    print('Classification report \n', classification_report(te_Y, pred, zero_division=0))


    
