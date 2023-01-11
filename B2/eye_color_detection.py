import os
from B2.feature_extraction import extract_features_labels
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
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

    print(tr_X.shape, te_X.shape)

    c_values = np.logspace(-3, 1, 5)

    params = {
        'C':c_values 
    }

    clf = LogisticRegression(multi_class = 'multinomial', solver = 'lbfgs')

    grid = GridSearchCV(clf, param_grid=params, verbose=3, refit = True)
    grid.fit(tr_X, tr_Y)

    pred = grid.predict(te_X)

    print("SoftMax Accuracy:", accuracy_score(te_Y, pred))
    print('SoftMax Classification report \n', classification_report(te_Y, pred, zero_division=0))


def test_eye_color_detection():
    tr_X, tr_Y, te_X, te_Y = get_data()

    tr_X = tr_X.reshape((tr_X.shape[0], tr_X.shape[1]*tr_X.shape[2]*tr_X.shape[3]))
    te_X = te_X.reshape((te_X.shape[0], te_X.shape[1]*te_X.shape[2]*te_X.shape[3]))

    predict(tr_X, tr_Y, te_X, te_Y)