import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score

from data_setup import create_dataset, split_datasets, load_imdb_dataset, dataset_metrics
from extract_features import featurize_dataset
from model_training import model_training, get_model_predictions
from calibration import plot_calibration_curve, calibrate_temperature_scaling, TemperatureScaling, calibrate_histogram_binning, calibrate_isotonic_regression, calibrate_beta_calibration
from classifiers import TextClassificationModel

# constants
labeled_percentage = 0.1 # percentage of training data to use as initial set of labeled data (for training)
validation_percentage = 0.45 # percentage of training data to use as validation set for determining calibration parameters
batch_size = 64
threshold = 0.8 # used to determine which unlabeled examples have high enough confidence
num_classes = 2 
num_epochs = 10
num_self_training_iterations = 1000000

criterion = nn.CrossEntropyLoss() # what about log loss?

def test_calibration(calibration_method, folder_name):
    # reproducible
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train, unlabeled, test = load_imdb_dataset('../')

    # for testing purposes
    # train = train[0][:100], train[1][:100]
    # unlabeled = unlabeled[0][:100], unlabeled[1][:100]
    # test = test[0][:100], test[1][:100]

    # split datasets to get validation and updated train and unlabeled sets
    (train_text, train_labels), (validation_text, validation_labels), (test_text, test_labels), (unlabeled_text, unlabeled_labels) = split_datasets(train, test, unlabeled, 0.1, 0.9)

    # create dataset objects for each split
    train_dataset = create_dataset(train_text, train_labels)
    validation_dataset = create_dataset(validation_text, validation_labels)
    test_dataset = create_dataset(test_text, test_labels)

    # extract features and create dataloader for each split
    train_dataloader = featurize_dataset(train_dataset, device, batch_size)
    test_dataloader = featurize_dataset(test_dataset, device, batch_size)
    validation_dataloader = featurize_dataset(validation_dataset, device, batch_size)

    model = TextClassificationModel(768, 2)
    model = model_training(model, device, 100, train_dataloader, criterion)

    methods = {'histogram binning': calibrate_histogram_binning, 'isotonic regression': calibrate_isotonic_regression, 'beta calibration': calibrate_beta_calibration, 'temperature scaling': calibrate_temperature_scaling}
    calibration_class = methods[calibration_method](validation_dataloader, device, model)

    if calibration_method == 'temperature scaling':
        print('T for temperature scaling')
        print(calibration_class.temp)

    # examine calibration on train set
    train_pre_softmax_probs, train_post_softmax_probs, train_predicted_probs, train_predicted_labels, train_true_labels = get_model_predictions(train_dataloader, device, model)
    train_accuracy = accuracy_score(train_true_labels, train_predicted_labels)
    print('Train accuracy: ', train_accuracy)

    plot_calibration_curve(train_true_labels, train_post_softmax_probs, folder_name + '/temp_scaling_train_initial_calibration.jpg')

    # recalibrate and examine new calibration on train set
    if calibration_method == 'temperature scaling':
        calibrated_train_probs = calibration_class.predict(train_pre_softmax_probs)
    else:
        calibrated_train_probs = calibration_class.predict(train_predicted_probs)
    plot_calibration_curve(train_true_labels, calibrated_train_probs, folder_name + '/temp_scaling_train_after_calibration.jpg')

    # examine calibration on test set
    test_pre_softmax_probs, test_post_softmax_probs, test_predicted_probs, test_predicted_labels, test_true_labels = get_model_predictions(test_dataloader, device, model)
    test_accuracy = accuracy_score(test_true_labels, test_predicted_labels)
    print('Test accuracy: ', test_accuracy)

    plot_calibration_curve(test_true_labels, test_post_softmax_probs, folder_name + '/temp_scaling_test_initial_calibration.jpg')

    # recalibrate and examine new calibration on test set
    if calibration_method == 'temperature scaling':
        calibrated_test_probs = calibration_class.predict(test_pre_softmax_probs)
    else:
        calibrated_test_probs = calibration_class.predict(test_predicted_probs)
    plot_calibration_curve(test_true_labels, calibrated_test_probs, folder_name + '/temp_scaling_test_after_calibration.jpg')

def main(model, criterion):

    # reproducible
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train, unlabeled, test = load_imdb_dataset('../')

    # split datasets to get validation and updated train and unlabeled sets
    (train_text, train_labels), (validation_text, validation_labels), (test_text, test_labels), (unlabeled_text, unlabeled_labels) = split_datasets(train, test, unlabeled, labeled_percentage, validation_percentage)

    # create dataset objects for each split
    train_dataset = create_dataset(train_text, train_labels)
    validation_dataset = create_dataset(validation_text, validation_labels)
    test_dataset = create_dataset(test_text, test_labels)
    unlabeled_dataset = create_dataset(unlabeled_text, unlabeled_labels)

    # extract features and create dataloader for each split
    train_dataloader = featurize_dataset(train_dataset, device, batch_size)
    test_dataloader = featurize_dataset(test_dataset, device, batch_size)
    unlabeled_dataloader = featurize_dataset(unlabeled_dataset, device, batch_size)
    validation_dataloader = featurize_dataset(validation_dataset, device, batch_size)

    # metrics 
    accuracy = []
    precision = []
    recall = []
    f1 = []
    auc_roc = []
    training_data_size = []
    expected_calibration_errors = []
    expected_calibration_errors_recalibration = []
    Ts = []

    for i in range(num_self_training_iterations):

        # ensure new model for each iteration
        model = model_training(model, device, num_epochs, train_dataloader, criterion)

        # predictions on test set
        pre_softmax_probs, post_softmax_probs, predicted_labels, true_labels = get_model_predictions(test_dataloader, device, model)

        # check calibration
        classifier_probs_all = post_softmax_probs
        classifier_probs = post_softmax_probs[:,1]
        c_prob_true, c_prob_pred = calibration_curve(true_labels, classifier_probs, n_bins=10) # import calibration curve from somewhere else???

        # plot calibration curve

        # update metrics

        # recalibrate 
        T = calibrate_temperature_scaling(validation_dataloader, device, model)

        # plot new calibration curve

        # update metrics for after recalibration

        # predictions on unlabeled
        pre_softmax_probs, post_softmax_probs, predicted_labels, true_labels = get_model_predictions(unlabeled_dataloader, device, model)

        # add new samples to training data based on unlabeled predictions

    # plot metrics

    # get/store metric values

print('temperature scaling')
test_calibration('temperature scaling', 'temperature_scaling_test6')
# print('histogram binning')
# test_calibration('histogram binning', 'temperature_scaling_test4')
# print('isotonic regression')
# test_calibration('isotonic regression', 'temperature_scaling_test4')
# print('beta calibration')
# test_calibration('beta calibration', 'temperature_scaling_test4')


