import numpy as np
import Model_Parameters as mp
from DatasetCrossValidation import DatasetCrossValidation
from trainModel import train_model, predict_label, evaluate_model

'''
Define parameters and Hyper-parameters of the model.
'''
params = mp.params()
train, test = mp.load_data()
train = mp.change_labels(train) # change lables to -1/1 for classification.

CV_model = np.zeros([params['CV_k'], train['X'].shape[1]])
precision = np.zeros([params['CV_k']])
convergenceVec = np.zeros([params['CV_k']])

'''
training with Cross-Validation
'''
for fold in range(params['CV_k']):

    # extract validation  and training set:
    validation_X, validation_Y, training_X, training_Y = DatasetCrossValidation(train, fold, params['CV_k'])

    # train model
    fold_model = train_model(training_X, training_Y, params)
    CV_model[fold, :] = fold_model['W']

    # predict
    prediction = predict_label(validation_X, CV_model[fold, :])

    # Evaluate model
    precision[fold] = evaluate_model(prediction.T, validation_Y)

    #create a vector of step until convergence at each fold.
    convergenceVec[fold] = fold_model['num_of_epochs']

meanPrecision = np.mean(precision) # save the mean precision across folds, to evaluate the expected accuracy.
meanSteps = np.mean(convergenceVec) # save the mean number of steps until convergence across folds.

'''
final model:
'''
# train model
final_model = train_model(train['X'], train['Y'], params)
# predict
prediction = predict_label(test, final_model['W'])

'''
results
'''
rightMove = np.where(prediction == 1)  # extract the indices of right move classification.
upMove = np.where(prediction == -1)  # extract the indices of up move classification.

'''
print success rate and other data of the classifier.
'''
print('Right movements are : ' + str(rightMove))
print('Up movements are : ' + str(upMove))
print('Expected accuracy over validation sets is : ' + str(meanPrecision) + ' , values : ' + str(precision))
print('Mean number of steps to convergence is : ' + str(meanSteps) + ' , values : ' + str(convergenceVec))
print('Mean steps with final is : '+str((np.sum(convergenceVec)+final_model['num_of_epochs'])/(len(convergenceVec)+1))
      + ' (final was ' + str(final_model['num_of_epochs']) + ')')

