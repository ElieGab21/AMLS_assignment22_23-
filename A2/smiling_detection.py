import A2.lab2_landmarks as l2
import numpy as np
from sklearn.metrics import classification_report,accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import os

def get_data():
    '''
    This function retrieves the data using the feature extraction function

    returns:
        tr_X, tr_Y, te_X, te_Y: The datasets 
    
    '''
        
    basedir = 'Datasets/celeba'
    base_test_dir = 'Datasets/celeba_test'
    img_path = os.path.join(basedir, 'img')
    img_test_path = os.path.join(base_test_dir, 'img')

    tr_X, tr_Y = l2.extract_features_labels(img_path, basedir)
    tr_Y = np.array([tr_Y, -(tr_Y - 1)]).T

    te_X, te_Y = l2.extract_features_labels(img_test_path, base_test_dir)
    te_Y = np.array([te_Y, -(te_Y - 1)]).T       

    return tr_X, tr_Y, te_X, te_Y

def first_model_pipeline(X_train, Y_train, X_test, Y_test):
    """
    This function tests 5 different models without any tuning

    Args:
        - X_train: Training X data
        - Y_train: Training Y data
        - X_test: testing X data
        - Y_test: testing Y data
    
    Returns:
        - accuracy_dict: A dictionnary containing the accuracy of each model after prediction
    """

    models_pipeline = [LogisticRegression(solver='liblinear'), SVC(kernel='linear'), SVC(kernel='poly'), KNeighborsClassifier(), DecisionTreeClassifier()]
    accuracy_dict = {}
    model_names = ['Log regression', 'SVM (Linear)', 'SVM (Poly)', 'K nearest Neighbor', 'Decision Tree']
    i = 0

    for model in models_pipeline:

        model.fit(X_train, Y_train)
        pred = model.predict(X_test)

        accuracy_dict[model_names[i]] = accuracy_score(Y_test, pred)

        i+=1

    return accuracy_dict

def model_tuning(X_train, Y_train, X_test, Y_test):
    """
    This function the 3 best models and tunes them using GridSearchCV

    Args:
        - X_train: Training X data
        - Y_train: Training Y data
        - X_test: testing X data
        - Y_test: testing Y data
    
    Returns:
        - accuracy_dict: A dictionnary containing the accuracy of each model after prediction
    """

    models_pipeline = [LogisticRegression(solver='liblinear'), SVC(kernel='linear'), SVC(kernel='poly')]    
    model_names = ['Log regression', 'SVM (Linear)', 'SVM (Poly)']
    accuracy_dict = {}

    c_values = np.logspace(-3, 1, 5)
    i = 0

    for model in models_pipeline:
        params = {
            'C':c_values 
        }

        #Testing regularization techniques (Ridge then Lasso)
        if model_names[i] == 'Log regression':
            params['penalty'] = ['l1', 'l2']

        #Testing different degrees for polynomial SVM
        elif model_names[i] == 'SVM (Poly)':
            params['degree'] = [1, 2, 3, 4]

        grid = GridSearchCV(model, param_grid=params, verbose=3, refit = True)
        grid.fit(X_train, Y_train)

        pred = grid.predict(X_test)

        accuracy_dict[model_names[i]] = [accuracy_score(Y_test, pred), grid.best_params_]

        i+=1

    return accuracy_dict


def predict(training_images, training_labels, test_images, test_labels):
    """
    This function calls the two testing functions, first_model_pipeline() and model_tuning()
    to test the different models. It then uses the best model to fit and predict the dataset

    Prints the accuracy score and classification report
    """

    #Scaling the data
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(training_images)
    X_test = scaler.transform(test_images)
    
    # Pipeline (uncomment lines 87-92 to test)
    # accuracies = first_model_pipeline(X_train, training_labels, X_test, test_labels)

    # new_accuracies = model_tuning(X_train, training_labels, X_test, test_labels)

    # print('First pipline:', accuracies)
    # print('After tuning:', new_accuracies)

    # Pipeline Results:
    # First pipline: {'Log regression': 0.9, 'SVM (Linear)': 0.9020618556701031, 'SVM (Poly)': 0.8907216494845361, 'K nearest Neighbor': 0.8680412371134021, 'Decision Tree': 0.8515463917525773}
    # After tuning: {'Log regression': [0.9030927835051547, {'C': 10.0, 'penalty': 'l1'}], 'SVM (Linear)': [0.9041237113402062, {'C': 10.0}], 'SVM (Poly)': [0.8907216494845361, {'C': 1.0, 'degree': 3}]}   

    # Fitting the best result
    classifier = SVC(kernel='linear', C = 10)

    classifier.fit(X_train, training_labels)

    pred = classifier.predict(X_test)

    print("Accuracy:", accuracy_score(test_labels, pred))
    print('Classification report \n', classification_report(test_labels, pred, zero_division=0))

def test_smile_detection():
    tr_X, tr_Y, te_X, te_Y= get_data()
    pred=predict(tr_X.reshape((tr_X.shape[0], 68*2)), list(zip(*tr_Y))[0], te_X.reshape((te_X.shape[0], 68*2)), list(zip(*te_Y))[0])