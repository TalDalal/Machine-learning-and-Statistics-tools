import numpy as np

def DatasetCrossValidation(data, fold, out_of):
    '''
    Extract data for cross-validation
    '''

    num_samples = data['X'].shape[0]
    chunk_size = num_samples / out_of
    surplus = num_samples % chunk_size
    strtidx = range(0, num_samples, chunk_size)
    print(strtidx)
    for i in range(surplus, 0, -1):
        strtidx[-i] += surplus - i + 1
    first = strtidx[fold]
    last = strtidx[fold + 1]

    '''
    Extract validation set and remaining data as training set
    '''

    validation_X = data['X'][first:last, :]
    validation_Y = data['Y'][first:last]

    training_Y = np.array(np.concatenate((data['Y'][0:first, :], data['Y'][last:-1, :]), axis=0))
    training_X = np.array(np.concatenate((data['X'][0:first, :], data['X'][last:-1, :]), axis=0))


    return validation_X, validation_Y, training_X, training_Y