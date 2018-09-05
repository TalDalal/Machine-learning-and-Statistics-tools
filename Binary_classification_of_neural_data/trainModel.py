import random
import numpy as np
import matplotlib.pyplot as plt



def train_model(data_X, data_Y, params):
    random.seed(params['seed'])
    np.random.seed(params['seed'])

    num_samples, num_features = data_X.shape

    # initialize W to random values
    W=np.zeros(num_features)
    for i in range(num_features):

        W[i] = random.uniform(0, 1)

    '''
    SGD with hinge - loss
    '''
    error = np.zeros([params['max_epoch']])
    for epoch in range(params['max_epoch']):
        print('\nEpoch #%i: ' % epoch)

        # Arrange samples in random order for each learning epoch
        epoch_order = np.random.permutation(num_samples)
        error[epoch] = 0

        for iteration in range(num_samples):

            # Get current sample
            sample_index = epoch_order[iteration]
            curr_sample = data_X[sample_index,:]
            curr_label = data_Y[sample_index,:]

            # Update weights
            condition = np.dot(np.dot(W, curr_sample.T), curr_label)
            if condition < 0:

                W += np.dot(params['eta'],curr_sample)*curr_label
                error[epoch] += -condition

                # Plot average error
                if epoch > params['convergence_window']:
                    error_val = np.mean(error[epoch-params['convergence_window']:epoch])
                    plt.plot(epoch, error_val, 'ro')


        '''
        Stopping criteria
        '''
        if error[epoch] < params['threshold']:
            break

    plt.title('error over epochs')
    plt.xlabel('learning epoch')
    plt.ylabel('train error')
    plt.show()


    if epoch < params['max_epoch']:
        print('\nStopped after %i epochs\n' % epoch)
    else:
        print('\nNo convergence - epoch number exceeded maximal number of epochs\n')

    # Output model
    model = {'W': W, 'training_error': error, 'num_of_epochs': epoch}

    return model


def predict_label(X, W):
        predicted = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            if np.sign(np.dot(X[i],W)) == 1:
               predicted[i] = 1
            else:
               predicted[i] = -1
        return predicted


def evaluate_model(predicted_y, true_y):
    c = 0
    for i in range(predicted_y.shape[0]):
        if predicted_y[i]==true_y[i]:
            c += 1

    precision = 100*(c/float(np.shape(predicted_y)[0]))

    return precision
