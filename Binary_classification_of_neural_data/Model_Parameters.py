import scipy.io as sp
import os
import numpy as np

def params():
    '''
    :returns Parameters & Hyper-Parameters of the model.
    '''
    params = {'seed': 1, 'eta': 0.1, 'max_epoch': 10000, 'convergence_window': 50, 'CV_k': 5, 'threshold': 100000}
    return params


def load_data():
    '''
    loads the neural data from MATLAB file
    :returns training set containing vectors of features and labels,
             a test set containing features only
    '''
    dataset_file = 'dataset_SMA.mat'
    dir_path = os.path.dirname(os.path.realpath(__file__))
    full_path = dir_path + '/data/' + dataset_file
    mat = sp.loadmat(full_path)
    test = mat['data_test'][0, 0]['X']
    train = mat['data_train'][0, 0]
    return train, test


def change_labels(train):
    '''
    :return nueral response for moving the hand up vs right.
    the label for 'up' was 7, so this function changes the labels to '-1', and right remains +1.
    '''
    train['Y'] = train['Y'].astype(np.int8)
    train['Y'][train['Y'] == 7] = -1
    assert train['Y'].any() == 1 or train['Y'].any() == -1 , 'labels not changed correctly'
    return train