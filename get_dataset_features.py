import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from data_setup import create_dataset, split_datasets, load_imdb_dataset, load_sst2_dataset, load_sst5_dataset, load_amazon_elec_dataset, load_amazon_elec_binary_dataset, load_dbpedia_dataset, load_ag_news_dataset, load_yelp_full_dataset, load_yelp_polarity_dataset, load_amazon_full_dataset, load_amazon_polarity_dataset, load_yahoo_answers_dataset, load_twenty_news_dataset, load_airport_tweets_dataset, dataset_metrics, TextDataset, get_dataset_from_dataloader
from extract_features import featurize_dataset
from model_training import model_training, get_model_predictions
from calibration import calibrate_platt_scaling, plot_calibration_curve, calibrate_temperature_scaling, calibrate_histogram_binning, calibrate_isotonic_regression, calibrate_beta_calibration, calibrate_equal_freq_binning, calibrate_bbq, calibrate_ensemble_temperature_scaling, calibrate_enir, calibrate_platt_binner
from selecting_unlabeled import unlabeled_samples_to_train
from classifiers import TextClassificationModel

# constants
labeled_percentage = 0.2 # percentage of training data to use as initial set of labeled data (for training)
validation_percentage = 0.1 # percentage of training data to use as validation set for determining calibration parameters
validation_model_percentage = 0.2 # percentage of training data to use as validation set for tuning model
batch_size = 256
threshold = 0.8 # used to determine which unlabeled examples have high enough confidence
num_classes = 2 
num_epochs = 10
num_self_training_iterations = 1000000

criterion = nn.CrossEntropyLoss() 

def main(models, dataset, criterion, recalibration_method, folder_name, load_features = False, calibrate = True, label_smoothing = 'none', label_smoothing_alpha=None, retrain_models_from_scratch=True, label_smoothing_model_training=False, label_smoothing_model_training_alpha=0.1, k_best=None, k=None, k_best_and_threshold=None):

    # reproducible
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if load_features: 
        train_features = torch.load(dataset + '_features_train_data.pt')
        train_labels = torch.load(dataset + '_labels_train_data.pt')

        test_features = torch.load(dataset + '_features_test_data.pt')
        test_labels = torch.load(dataset + '_labels_test_data.pt')

        unlabeled_features = torch.load(dataset + '_features_unlabeled_data.pt')
        unlabeled_labels = torch.load(dataset + '_labels_unlabeled_data.pt')

        (train_features, train_labels), (validation_features, validation_labels), (test_features, test_labels), (unlabeled_features, unlabeled_labels) = split_datasets((train_features, train_labels), test=(test_features, test_labels), unlabeled=(unlabeled_features, unlabeled_labels), labeled_proportion=labeled_percentage, validation_proportion=validation_percentage)

        train_dataset_size = len(train_labels)
        train_dataloader = DataLoader(TextDataset(train_features, train_labels), batch_size=batch_size)
        test_dataloader = DataLoader(TextDataset(test_features, test_labels), batch_size=batch_size)
        validation_dataloader = DataLoader(TextDataset(validation_features, validation_labels), batch_size=batch_size)
        unlabeled_dataloader = DataLoader(TextDataset(unlabeled_features, unlabeled_labels), batch_size=batch_size)
    else:
        dataset_name_to_load_func = {'imdb': load_imdb_dataset, 'sst2': load_sst2_dataset, 'sst5': load_sst5_dataset, 'amazon_elec': load_amazon_elec_dataset, 'amazon_elec_binary': load_amazon_elec_binary_dataset, 'dbpedia': load_dbpedia_dataset, 'ag_news': load_ag_news_dataset, 'yelp_full': load_yelp_full_dataset, 'yelp_polarity': load_yelp_polarity_dataset, 'amazon_full': load_amazon_full_dataset, 'amazon_polarity': load_amazon_polarity_dataset, 'yahoo': load_yahoo_answers_dataset, 'twenty_news': load_twenty_news_dataset, 'airport_tweets': load_airport_tweets_dataset}

        data = dataset_name_to_load_func[dataset]('../')

        # split data into train, unlabeled, test
        if len(data) == 1:
            train = data[0]
            test = None 
            unlabeled = None
        elif len(data) == 2:
            train, test = data
            unlabeled = None
        else:
            train, unlabeled, test = data

        # for testing purposes
        # train = train[0][:100], train[1][:100]
        # unlabeled = unlabeled[0][:100], unlabeled[1][:100]
        # test = test[0][:100], test[1][:100]

        # create dataset objects for each split
        train_dataset = create_dataset(train[0], train[1])
        if test is not None:
            test_dataset = create_dataset(test[0], test[1])
        if unlabeled is not None:
            unlabeled_dataset = create_dataset(unlabeled[0], unlabeled[1])

        # extract features 
        if dataset != 'sst2' and dataset != 'sst5' and dataset != 'airport_tweets': # TEMPORARY
            train = featurize_dataset(train_dataset, device, batch_size, dataset, 'train_data.pt')
        else:
            train_features = torch.load(dataset + '_features_train_data.pt')
            train_labels = torch.load(dataset + '_labels_train_data.pt')
            train = train_features, train_labels
        if test is not None:
            test = featurize_dataset(test_dataset, device, batch_size, dataset, 'test_data.pt')
        if unlabeled is not None:
            unlabeled = featurize_dataset(unlabeled_dataset, device, batch_size, dataset, 'unlabeled_data.pt')

        # split datasets to get validation and updated train and unlabeled sets
        (train_features, train_labels), (validation_features, validation_labels), (test_features, test_labels), (unlabeled_features, unlabeled_labels) = split_datasets(train, test=test, unlabeled=unlabeled, labeled_proportion=labeled_percentage, validation_proportion=validation_percentage)

        train_dataset_size = len(train_labels)

        # create dataloaders 
        train_dataloader = DataLoader(TextDataset(train_features, train_labels), batch_size=batch_size)
        test_dataloader = DataLoader(TextDataset(test_features, test_labels), batch_size=batch_size)
        validation_dataloader = DataLoader(TextDataset(validation_features, validation_labels), batch_size=batch_size)
        unlabeled_dataloader = DataLoader(TextDataset(unlabeled_features, unlabeled_labels), batch_size=batch_size)

    # metrics - also generate plots w/ probabilty distributions and calibration curves for each iteration
    accuracy = []
    precision = []
    recall = []
    f1 = []
    auc_roc = []
    training_data_size = []
    expected_calibration_errors = []
    expected_calibration_errors_recalibration = []

    all_unlabeled_data_used = False

    for i in range(num_self_training_iterations):

        # ensure new model for each iteration
        for j in range(len(models)):
            models[j] = model_training(models[j], device, num_epochs, train_dataloader, criterion, label_smoothing=label_smoothing_model_training, label_smoothing_alpha=label_smoothing_model_training_alpha)

        # predictions on test set
        pre_softmax_probs, post_softmax_probs, predicted_probs, predicted_labels, true_labels, _ = get_model_predictions(test_dataloader, device, models, use_pre_softmax=False, use_post_softmax=True)

        # check calibration
        ece, _, _, _, _ = plot_calibration_curve(true_labels, post_softmax_probs, folder_name + '/' + dataset + '_' + recalibration_method + '_iteration' + str(i) + '_test_initial_calibration.jpg')

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
            methods = {'histogram_binning': calibrate_histogram_binning, 'isotonic_regression': calibrate_isotonic_regression, 'beta_calibration': calibrate_beta_calibration, 'temp_scaling': calibrate_temperature_scaling, 'platt_scaling': calibrate_platt_scaling, 'equal_freq_binning': calibrate_equal_freq_binning, 'bbq': calibrate_bbq, 'ensemble_temperature_scaling': calibrate_ensemble_temperature_scaling, 'enir': calibrate_enir, 'platt_binning': calibrate_platt_binner}
            calibration_class = methods[recalibration_method]

            if recalibration_method in ('temp_scaling', 'platt_scaling', 'ensemble_temperature_scaling'):
                calibrated_probs = calibration_class(validation_dataloader, device, models, pre_softmax_probs, label_smoothing, label_smoothing_alpha)
            else:
                calibrated_probs = calibration_class(validation_dataloader, device, models, post_softmax_probs, label_smoothing, label_smoothing_alpha)

            # plot new calibration curve
            ece, _, _, _, _ = plot_calibration_curve(true_labels, calibrated_probs, folder_name + '/' + dataset + '_' + recalibration_method + '_iteration' + str(i) + '_test_after_calibration.jpg')

            # update metrics for after recalibration
            expected_calibration_errors_recalibration.append(ece)
        else:
            calibration_class = None

        if all_unlabeled_data_used:
            print('No more unlabeled data')
            print('Exited on iteration ', i)
            break

        train_dataloader, unlabeled_dataloader, num_samples_added_to_train, num_unlabeled_samples = unlabeled_samples_to_train(models, device, train_dataloader, validation_dataloader, unlabeled_dataloader, calibrate, recalibration_method, calibration_class, label_smoothing, threshold, batch_size, k_best=None, k=None, k_best_and_threshold=None)


        train_dataset_size += num_samples_added_to_train

        # check for end conditions (no more unlabeled data OR no new samples added to training)
        if num_samples_added_to_train == num_unlabeled_samples: # all unlabeled samples used, will exit in next iteration
            all_unlabeled_data_used = True
        if num_samples_added_to_train == 0:
            print('No more samples with high enough confidence')
            print('Exited on iteration ', i)
            break

        # reset models to re-train from scratch in next iteration
        if retrain_models_from_scratch:
            for model in models:
                for layer in model.children():
                    if hasattr(layer, 'reset_parameters'):
                        layer.reset_parameters()

    # plot metrics
    plt.figure()
    plt.plot(accuracy, color='red', label='accuracy')
    plt.plot(precision, color='blue', label='precision')
    plt.plot(recall, color='green', label='recall')
    plt.plot(f1, color='orange', label='f1')
    plt.plot(auc_roc, color='purple', label='auc_roc')
    plt.plot(expected_calibration_errors, color='yellow', label='expected_calibration')
    if calibrate:
        plt.plot(expected_calibration_errors_recalibration, color='pink', label='expected_calibration_recalibration')
    plt.legend()
    plt.savefig(folder_name + '/' + dataset + '_' + recalibration_method + '_metrics.jpg')

    plt.figure()
    plt.plot(training_data_size)
    plt.savefig(folder_name + '/' + dataset + '_' + recalibration_method + '_trainingdatasize.jpg')

    # get/store metric values
    metrics_dict = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'auc-roc': auc_roc, 'ece': expected_calibration_errors, 'training_data_size': training_data_size}
    if calibrate:
        metrics_dict['ece_after_recalibration'] = expected_calibration_errors_recalibration
    metrics = pd.DataFrame(data=metrics_dict)
    metrics.to_csv(folder_name + '/' + dataset + '_' + recalibration_method + '_metrics.csv')

model = TextClassificationModel(768, 2)

print('ag_news')
main([model], 'ag_news', criterion, 'temp_scaling', 'self_training_test13', load_features = False, calibrate = False)