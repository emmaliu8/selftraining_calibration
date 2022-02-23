import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from data_setup import create_dataset, split_datasets, load_imdb_dataset, dataset_metrics, TextDataset, get_dataset_from_dataloader
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

def test_calibration(calibration_method, folder_name, load_model = False, load_model_path = None, load_features = False):
    # reproducible
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if load_features:
        train_features = torch.load('features_train_data_testing_calibration.pt')
        train_labels = torch.load('labels_train_data_testing_calibration.pt')

        test_features = torch.load('features_test_data_testing_calibration.pt')
        test_labels = torch.load('labels_test_data_testing_calibration.pt')

        validation_features = torch.load('features_validation_data_testing_calibration.pt')
        validation_labels = torch.load('labels_validation_data_testing_calibration.pt')

        train_dataloader = DataLoader(TextDataset(train_features, train_labels), batch_size=batch_size)
        test_dataloader = DataLoader(TextDataset(test_features, test_labels), batch_size=batch_size)
        validation_dataloader = DataLoader(TextDataset(validation_features, validation_labels), batch_size=batch_size)
    else:
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
        train_dataloader = featurize_dataset(train_dataset, device, batch_size, 'train_data_testing_calibration.pt')
        test_dataloader = featurize_dataset(test_dataset, device, batch_size, 'test_data_testing_calibration.pt')
        validation_dataloader = featurize_dataset(validation_dataset, device, batch_size, 'validation_data_testing_calibration.pt')

    if load_model:
        model = TextClassificationModel(768, 2)
        model.load_state_dict(torch.load(load_model_path))
        model.to(device)
    else:
        model = TextClassificationModel(768, 2)
        model = model_training(model, device, 100, train_dataloader, criterion, 'test_calibration_model.pt')

    methods = {'histogram_binning': calibrate_histogram_binning, 'isotonic_regression': calibrate_isotonic_regression, 'beta_calibration': calibrate_beta_calibration, 'temp_scaling': calibrate_temperature_scaling}
    calibration_class = methods[calibration_method]

    # examine calibration on train set
    train_pre_softmax_probs, train_post_softmax_probs, train_predicted_probs, train_predicted_labels, train_true_labels = get_model_predictions(train_dataloader, device, model)
    train_accuracy = accuracy_score(train_true_labels, train_predicted_labels)
    print('Train accuracy: ', train_accuracy)

    plot_calibration_curve(train_true_labels, train_post_softmax_probs, folder_name + '/' + calibration_method + '_train_initial_calibration.jpg')

    # recalibrate and examine new calibration on train set
    if calibration_method == 'temp_scaling':
        calibrated_train_probs = calibration_class(validation_dataloader, device, model, train_pre_softmax_probs)
    else:
        calibrated_train_probs = calibration_class(validation_dataloader, device, model, train_post_softmax_probs)

    plot_calibration_curve(train_true_labels, calibrated_train_probs, folder_name + '/' + calibration_method + '_train_after_calibration.jpg')

    # examine calibration on test set
    test_pre_softmax_probs, test_post_softmax_probs, test_predicted_probs, test_predicted_labels, test_true_labels = get_model_predictions(test_dataloader, device, model)
    test_accuracy = accuracy_score(test_true_labels, test_predicted_labels)
    print('Test accuracy: ', test_accuracy)

    plot_calibration_curve(test_true_labels, test_post_softmax_probs, folder_name + '/' + calibration_method + '_test_initial_calibration.jpg')

    # recalibrate and examine new calibration on test set
    if calibration_method == 'temp_scaling':
        calibrated_test_probs = calibration_class(validation_dataloader, device, model, test_pre_softmax_probs)
    else:
        calibrated_test_probs = calibration_class(validation_dataloader, device, model, test_post_softmax_probs)

    plot_calibration_curve(test_true_labels, calibrated_test_probs, folder_name + '/' + calibration_method + '_test_after_calibration.jpg')

def main(model, criterion, recalibration_method, folder_name, load_features = False, calibrate = True):

    # reproducible
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if load_features: # need to add unlabeled
        train_features = torch.load('features_train_data_testing_calibration.pt')
        train_labels = torch.load('labels_train_data_testing_calibration.pt')
        train_dataset_size = len(train_labels)

        test_features = torch.load('features_test_data_testing_calibration.pt')
        test_labels = torch.load('labels_test_data_testing_calibration.pt')

        validation_features = torch.load('features_validation_data_testing_calibration.pt')
        validation_labels = torch.load('labels_validation_data_testing_calibration.pt')

        unlabeled_features = torch.load('features_unlabeled_data.pt')
        unlabeled_labels = torch.load('labels_unlabeled_data.pt')

        train_dataloader = DataLoader(TextDataset(train_features, train_labels), batch_size=batch_size)
        test_dataloader = DataLoader(TextDataset(test_features, test_labels), batch_size=batch_size)
        validation_dataloader = DataLoader(TextDataset(validation_features, validation_labels), batch_size=batch_size)
        unlabeled_dataloader = DataLoader(TextDataset(unlabeled_features, unlabeled_labels), batch_size=batch_size)
    else:
        train, unlabeled, test = load_imdb_dataset('../')

        # for testing purposes
        # train = train[0][:100], train[1][:100]
        # unlabeled = unlabeled[0][:100], unlabeled[1][:100]
        # test = test[0][:100], test[1][:100]

        # split datasets to get validation and updated train and unlabeled sets
        (train_text, train_labels), (validation_text, validation_labels), (test_text, test_labels), (unlabeled_text, unlabeled_labels) = split_datasets(train, test, unlabeled, labeled_percentage, validation_percentage)

        train_dataset_size = len(train_labels)

        # create dataset objects for each split
        train_dataset = create_dataset(train_text, train_labels)
        validation_dataset = create_dataset(validation_text, validation_labels)
        test_dataset = create_dataset(test_text, test_labels)
        unlabeled_dataset = create_dataset(unlabeled_text, unlabeled_labels)

        # extract features and create dataloader for each split
        train_dataloader = featurize_dataset(train_dataset, device, batch_size)
        test_dataloader = featurize_dataset(test_dataset, device, batch_size)
        unlabeled_dataloader = featurize_dataset(unlabeled_dataset, device, batch_size, 'unlabeled_data.pt')
        validation_dataloader = featurize_dataset(validation_dataset, device, batch_size)

    # metrics - also generate plots w/ probabilty distributions and calibration curves for each iteration
    accuracy = []
    precision = []
    recall = []
    f1 = []
    auc_roc = []
    training_data_size = []
    expected_calibration_errors = []
    expected_calibration_errors_recalibration = []

    for i in range(num_self_training_iterations):

        # ensure new model for each iteration
        model = model_training(model, device, num_epochs, train_dataloader, criterion)

        # predictions on test set
        pre_softmax_probs, post_softmax_probs, predicted_probs, predicted_labels, true_labels = get_model_predictions(test_dataloader, device, model)

        # check calibration
        ece, _, _, _, _ = plot_calibration_curve(true_labels, post_softmax_probs, folder_name + '/' + recalibration_method + '_iteration' + str(i) + '_test_initial_calibration.jpg')

        # update metrics
        accuracy.append(accuracy_score(true_labels, predicted_labels))
        precision.append(precision_score(true_labels, predicted_labels))
        recall.append(recall_score(true_labels, predicted_labels))
        f1.append(f1_score(true_labels, predicted_labels))
        auc_roc.append(roc_auc_score(true_labels, predicted_probs))
        training_data_size.append(train_dataset_size)
        expected_calibration_errors.append(ece)

        # recalibrate 
        if calibrate:
            methods = {'histogram_binning': calibrate_histogram_binning, 'isotonic_regression': calibrate_isotonic_regression, 'beta_calibration': calibrate_beta_calibration, 'temp_scaling': calibrate_temperature_scaling}
            calibration_class = methods[recalibration_method]

            if recalibration_method == 'temp_scaling':
                calibrated_probs = calibration_class(validation_dataloader, device, model, pre_softmax_probs)
            else:
                calibrated_probs = calibration_class(validation_dataloader, device, model, post_softmax_probs)

            # plot new calibration curve
            ece, _, _, _, _ = plot_calibration_curve(true_labels, calibrated_probs, folder_name + '/' + recalibration_method + '_iteration' + str(i) + '_test_after_calibration.jpg')

            # update metrics for after recalibration
            expected_calibration_errors_recalibration.append(ece)
        else:
            calibration_class = None

        def unlabeled_samples_to_train(train_dataloader, unlabeled_dataloader, recalibration_method, calibration_class):
            unlabeled_texts = [] # inputs that will remain in unlabeled set
            unlabeled_to_train_texts = []
            unlabeled_to_train_labels = []     

            model.eval()

            with torch.no_grad():
                for (_, batch) in enumerate(unlabeled_dataloader):
                    inputs = batch['Text'].to(device)

                    pre_softmax = model(inputs) 

                    # Get softmax values for net input and resulting class predictions
                    sm = nn.Softmax(dim=1)
                    post_softmax = sm(pre_softmax)

                    predicted_prob, predicted_label = torch.max(post_softmax.data, 1)
                    if calibrate:
                        if recalibration_method == 'temp_scaling':
                            unlabeled_calibrated_probs = calibration_class(validation_dataloader, device, model, pre_softmax.cpu().numpy())

                        else:
                            unlabeled_calibrated_probs = calibration_class(validation_dataloader, device, model, post_softmax.cpu().numpy())
 
                        unlabeled_calibrated_probs = np.max(unlabeled_calibrated_probs, axis=1)
                    
                    unlabeled_inputs = pd.DataFrame(np.array(inputs.cpu()))

                    df = pd.DataFrame([])
                    df['predicted_labels'] = predicted_label.cpu()

                    if calibrate:
                        df['predicted_probs'] = unlabeled_calibrated_probs
                    else:
                        df['predicted_probs'] = predicted_prob.cpu()

                    high_prob = df.loc[df['predicted_probs'] > threshold]
                    temp = np.array(unlabeled_inputs.drop(index=high_prob.index))

                    unlabeled_texts.append(temp)
                    temp2 = np.array(unlabeled_inputs.loc[high_prob.index])

                    unlabeled_to_train_texts.append(temp2)
                    temp3 = df['predicted_labels'].loc[high_prob.index]

                    unlabeled_to_train_labels.extend(df['predicted_labels'].loc[high_prob.index].tolist())


            num_samples_added_to_train = len(unlabeled_to_train_labels)
            unlabeled_to_train_labels = np.array(unlabeled_to_train_labels)
            unlabeled_to_train_texts = np.concatenate(unlabeled_to_train_texts)
            unlabeled_texts = np.concatenate(unlabeled_texts)
            num_unlabeled_samples = unlabeled_texts.shape[0]

            unlabeled_dataloader = DataLoader(TextDataset(unlabeled_texts, np.full((unlabeled_texts.shape[0], 1), -1)), batch_size=batch_size)

            train_text, train_labels = get_dataset_from_dataloader(train_dataloader, device)
            train_text = np.concatenate((train_text, unlabeled_to_train_texts))
            train_labels.extend(unlabeled_to_train_labels)
            train_dataloader = DataLoader(TextDataset(train_text, train_labels), batch_size=batch_size)

            return train_dataloader, unlabeled_dataloader, num_samples_added_to_train, num_unlabeled_samples
        
        train_dataloader, unlabeled_dataloader, num_samples_added_to_train, num_unlabeled_samples = unlabeled_samples_to_train(train_dataloader, unlabeled_dataloader, recalibration_method, calibration_class)

        train_dataset_size += num_samples_added_to_train

        # check for end conditions (no more unlabeled data OR no new samples added to training)
        # if len(unlabeled_dataloader) == 0 -> break
        if num_unlabeled_samples == 0:
            print('No more unlabeled data')
            print('Exited on iteration ', i)
            break
        if num_samples_added_to_train == 0:
            print('No more samples with high enough confidence')
            print('Exited on iteration ', i)
            break

    # plot metrics
    plt.figure()
    plt.plot(accuracy, color='red', label='accuracy')
    plt.plot(precision, color='blue', label='precision')
    plt.plot(recall, color='green', label='recall')
    plt.plot(f1, color='orange', label='f1')
    plt.plot(auc_roc, color='purple', label='auc_roc')
    plt.plot(expected_calibration_errors, color='yellow', label='expected_calibration')
    plt.plot(expected_calibration_errors_recalibration, color='pink', label='expected_calibration_recalibration')
    plt.legend()
    plt.savefig(folder_name + '/' + recalibration_method + '_metrics.jpg')

    plt.figure()
    plt.plot(training_data_size)
    plt.savefig(folder_name + '/' + recalibration_method + '_trainingdatasize.jpg')

    # get/store metric values

# testing calibration
# print('temperature scaling')
# test_calibration('temp_scaling', 'temperature_scaling_test8', load_model = True, load_model_path = 'test_calibration_model.pt', load_features = True)
# print('histogram binning')
# test_calibration('histogram_binning', 'temperature_scaling_test8', load_model = True, load_model_path = 'test_calibration_model.pt', load_features = True)
# print('isotonic regression')
# test_calibration('isotonic_regression', 'temperature_scaling_test8', load_model = True, load_model_path = 'test_calibration_model.pt', load_features = True)
# print('beta calibration')
# test_calibration('beta_calibration', 'temperature_scaling_test8', load_model = True, load_model_path = 'test_calibration_model.pt', load_features = True)

model = TextClassificationModel(768, 2)
main(model, criterion, 'temp_scaling', 'self_training_test2', load_features = True)

