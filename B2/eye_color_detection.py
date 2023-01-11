import os
from B2.feature_extraction import extract_features_labels

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
    # te_X, te_Y = extract_features_labels(img_test_path, base_test_dir)

    te_X, te_Y = 0, 0

    return tr_X, tr_Y, te_X, te_Y

def test_eye_color_detection():
    tr_X, tr_Y, te_X, te_Y = get_data()

    print(tr_X.shape)