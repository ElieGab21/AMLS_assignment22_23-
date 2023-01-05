import A1.lab2_landmarks as l2
import numpy as np
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
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
    # Can use gridsearch to test multiple hyperparameters
    # https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/

    #Model
    classifier = SVC()
    # classifier.fit(training_images, training_labels)
    # pred = classifier.predict(test_images)

    #Finding the best parameters
    param_grid = {'C': [0.01, 0.1, 1, 10],
              'kernel': ['linear']} 

    grid = GridSearchCV(classifier, param_grid, refit = True, verbose = 3)

    #Scaling the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(training_images)
    X_test = scaler .transform(test_images)

    #Fitting and training the model with best parameters
    grid.fit(X_train, training_labels)

    pred = grid.predict(X_test)

    print("Accuracy:", accuracy_score(test_labels, pred))
    print(grid.best_params_)
    print('Classification report \n', classification_report(test_labels, pred, zero_division=0))

    return pred

def test_gender_detection():
    tr_X, tr_Y, te_X, te_Y= get_data()
    pred=img_SVM(tr_X.reshape((tr_X.shape[0], 68*2)), list(zip(*tr_Y))[0], te_X.reshape((te_X.shape[0], 68*2)), list(zip(*te_Y))[0])