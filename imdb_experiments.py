import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
import os

from data_setup import create_dataset, split_datasets, load_imdb_dataset, load_sst2_dataset, load_sst5_dataset, load_amazon_elec_dataset, load_amazon_elec_binary_dataset, load_modified_amazon_elec_binary_dataset, load_dbpedia_dataset, load_ag_news_dataset, load_yelp_full_dataset, load_yelp_polarity_dataset, load_amazon_full_dataset, load_amazon_polarity_dataset, load_yahoo_answers_dataset, load_twenty_news_dataset, load_airport_tweets_dataset, TextDataset
from extract_features import featurize_dataset, combine_dataset_files
from model_training import model_training, get_model_predictions
from calibration import calibrate_platt_scaling, plot_calibration_curve, calibrate_temperature_scaling, calibrate_histogram_binning, calibrate_isotonic_regression, calibrate_beta_calibration, calibrate_equal_freq_binning, calibrate_bbq, calibrate_ensemble_temperature_scaling, calibrate_enir, calibrate_platt_binner, calibrate_vector_scaling, calibrate_matrix_scaling
from selecting_unlabeled import unlabeled_samples_to_train
from classifiers import TextClassificationModel

# CONSTANTS
threshold = 0.8 # used to determine which unlabeled examples have high enough confidence
num_classes = 2
num_self_training_iterations = 1000000
batch_size = 64

criterion = nn.CrossEntropyLoss() 

def tune_model(labeled_percentage=0.2, # percentage of training data to use as initial set of labeled data (for training)
               model_validation_percentage=0.1, # percentage of training data to use as validation set for tuning model
               dataset = 'imdb', 
               dataset_has_test=True, # dataset has a given test set
               shared_validation=False # whether or not validation set for model tuning is the same as the validation set for calibration tuning (affects how the dataset splits are created)
               ):
    '''
    Outputs model accuracy after training with different hyperparameters
    '''
    parameters_to_tune = ['num_epochs', 'learning_rate', 'batch_size']
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load data
    train_features, train_labels = combine_dataset_files(dataset, 'train', '.')

    # get training data and validation data for tuning model from original train data
    model_validation_size = int(model_validation_percentage * len(train_labels))
    train_size = int(labeled_percentage * len(train_labels))
    test_size = int(0.2 * len(train_labels))
    if shared_validation:
        if dataset_has_test:
             model_train_features, model_train_labels = train_features[:train_size], train_labels[:train_size]
             model_validation_features, model_validation_labels = train_features[train_size: train_size + model_validation_size], train_labels[train_size: train_size + model_validation_size]
        else:
            model_train_features, model_train_labels = train_features[:train_size], train_labels[:train_size]
            model_validation_features, model_validation_labels = train_features[train_size + test_size: train_size + test_size + model_validation_size], train_labels[train_size + test_size: train_size + test_size + model_validation_size]
    else:
        model_validation_features, model_validation_labels = train_features[:model_validation_size], train_labels[:model_validation_size]
        model_train_features, model_train_labels = train_features[model_validation_size: model_validation_size + train_size], train_labels[model_validation_size: model_validation_size + train_size]
    
    results = pd.DataFrame(columns=['num_epochs', 'batch_size', 'learning_rate', 'accuracy'])

    # model tuning
    for epoch_value in [5, 10, 15, 20, 25, 30]:
        for batch_size_value in [64, 128, 256, 512]:
            for lr_value in [0.00001, 0.0001, 0.001, 0.01, 0.1]:
                train_dataloader = DataLoader(TextDataset(model_train_features, model_train_labels), batch_size=batch_size_value)
                validation_dataloader = DataLoader(TextDataset(model_validation_features, model_validation_labels), batch_size=batch_size_value)

                model = TextClassificationModel(768, 2)
                trained_model = model_training(model, device, train_dataloader, criterion, num_epochs = epoch_value, learning_rate=lr_value)

                # evaluate trained model on validation data 
                _, _, _, predicted_labels, true_labels, _ = get_model_predictions(validation_dataloader, device, [trained_model], use_pre_softmax=False, use_post_softmax=True)
                accuracy = accuracy_score(true_labels, predicted_labels)

                results.loc[len(results.index)] = [epoch_value, batch_size_value, lr_value, accuracy]

    results = results.sort_values(by=['accuracy'], ascending=False)
    results.to_csv('tuning_one_layer_on_' + dataset + '_' + str(labeled_percentage) + '_' + str(model_validation_percentage) + '_' + str(shared_validation) + '.csv') 

def test_calibration(calibration_method, 
                     folder_name, # for saving results 
                     load_model = False, # load previously trained model
                     load_model_path = None, 
                     load_features = False, # load previously computed features
                     label_smoothing = 'none' # 'none', 'alpha', or 'platt'
                     ):
    '''
    Tests calibration methods on the IMDB dataset
    '''
    # reproducible
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if load_features:
        train_features = torch.load('imdb_features_train_data.pt')
        train_labels = torch.load('imdb_labels_train_data.pt')

        test_features = torch.load('imdb_features_test_data.pt')
        test_labels = torch.load('imdb_labels_test_data.pt')

        # unlabeled_features = torch.load('features_unlabeled_data.pt')
        # unlabeled_labels = torch.load('labels_unlabeled_data.pt')

        (train_features, train_labels), validation, (test_features, test_labels), (unlabeled_features, unlabeled_labels) = split_datasets((train_features, train_labels), labeled_proportion=0.1, test=(test_features, test_labels), validation_proportion=0.1)

        if validation is not None:
            validation_features, validation_labels = validation
            validation_dataloader = DataLoader(TextDataset(validation_features, validation_labels), batch_size=batch_size)

        train_dataloader = DataLoader(TextDataset(train_features, train_labels), batch_size=batch_size)
        test_dataloader = DataLoader(TextDataset(test_features, test_labels), batch_size=batch_size)
        unlabeled_dataloader = DataLoader(TextDataset(unlabeled_features, unlabeled_labels), batch_size=batch_size)
    else:
        train, unlabeled, test = load_imdb_dataset('../')

        # for testing purposes
        # train = train[0][:100], train[1][:100]
        # unlabeled = unlabeled[0][:100], unlabeled[1][:100]
        # test = test[0][:100], test[1][:100]

        # create dataset objects for each split
        train_dataset = create_dataset(train[0], train[1])
        test_dataset = create_dataset(test[0], test[1])

        # extract features and create dataloader for each split
        train = featurize_dataset(train_dataset, device, batch_size, 'train_data.pt')
        test = featurize_dataset(test_dataset, device, batch_size, 'test_data.pt')

        # split datasets to get validation and updated train and unlabeled sets
        (train_features, train_labels), validation, (test_features, test_labels), (unlabeled_features, unlabeled_labels) = split_datasets(train, labeled_proportion=0.1,test=test, unlabeled=unlabeled, validation_proportion=0.9)

        if validation is not None:
            validation_features, validation_labels = validation
            validation_dataloader = DataLoader(TextDataset(validation_features, validation_labels), batch_size=batch_size)

        train_dataloader = DataLoader(TextDataset(train_features, train_labels), batch_size=batch_size)
        test_dataloader = DataLoader(TextDataset(test_features, test_labels), batch_size=batch_size)
        unlabeled_dataloader = DataLoader(TextDataset(unlabeled_features, unlabeled_labels), batch_size=batch_size)

    if load_model:
        model = TextClassificationModel(768, 2)
        model.load_state_dict(torch.load(load_model_path))
        model.to(device)
    else:
        model = TextClassificationModel(768, 5)
        model = model_training(model, device, train_dataloader, criterion, num_epochs=100, file_name='sst5_test_calibration_model.pt')

    methods = {'histogram_binning': calibrate_histogram_binning, 'isotonic_regression': calibrate_isotonic_regression, 'beta_calibration': calibrate_beta_calibration, 'temp_scaling': calibrate_temperature_scaling, 'platt_scaling': calibrate_platt_scaling, 'equal_freq_binning': calibrate_equal_freq_binning, 'bbq': calibrate_bbq, 'ensemble_temperature_scaling': calibrate_ensemble_temperature_scaling, 'enir': calibrate_enir, 'platt_binning': calibrate_platt_binner, 'vector_scaling': calibrate_vector_scaling, 'matrix_scaling': calibrate_matrix_scaling}
    calibration_class = methods[calibration_method]

    # examine calibration on train set
    train_pre_softmax_probs, train_post_softmax_probs, train_predicted_probs, train_predicted_labels, train_true_labels, _ = get_model_predictions(train_dataloader, device,[model])
    train_accuracy = accuracy_score(train_true_labels, train_predicted_labels)
    print('Train accuracy: ', train_accuracy)

    plot_calibration_curve(train_true_labels, train_post_softmax_probs, folder_name + '/' + calibration_method + '_train_initial_calibration.jpg')

    # recalibrate and examine new calibration on train set
    if calibration_method in ('temp_scaling', 'platt_scaling', 'ensemble_temperature_scaling'):
        calibrated_train_probs = calibration_class(validation_dataloader, device, [model], train_pre_softmax_probs, label_smoothing)
    elif calibration_method in ('vector_scaling', 'matrix_scaling'):
        calibrated_train_probs = calibration_class(validation_dataloader, device, [model], train_pre_softmax_probs, label_smoothing, num_classes=num_classes) 
    else:
        calibrated_train_probs = calibration_class(validation_dataloader, device, [model], train_post_softmax_probs, label_smoothing)

    plot_calibration_curve(train_true_labels, calibrated_train_probs, folder_name + '/' + calibration_method + '_train_after_calibration.jpg')

    # examine calibration on test set
    test_pre_softmax_probs, test_post_softmax_probs, test_predicted_probs, test_predicted_labels, test_true_labels, _ = get_model_predictions(test_dataloader, device, [model])
    test_accuracy = accuracy_score(test_true_labels, test_predicted_labels)
    print('Test accuracy: ', test_accuracy)

    plot_calibration_curve(test_true_labels, test_post_softmax_probs, folder_name + '/' + calibration_method + '_test_initial_calibration.jpg')

    # recalibrate and examine new calibration on test set
    if calibration_method in ('temp_scaling', 'platt_scaling', 'ensemble_temperature_scaling'):
        calibrated_test_probs = calibration_class(validation_dataloader, device, [model], test_pre_softmax_probs, label_smoothing)
    elif calibration_method in ('vector_scaling', 'matrix_scaling'):
        calibrated_test_probs = calibration_class(validation_dataloader, device, [model], test_pre_softmax_probs, label_smoothing, num_classes=num_classes)
    else:
        calibrated_test_probs = calibration_class(validation_dataloader, device, [model], test_post_softmax_probs, label_smoothing)

    plot_calibration_curve(test_true_labels, calibrated_test_probs, folder_name + '/' + calibration_method + '_test_after_calibration.jpg')

def running_test_calibration():
    # testing all calibration methods
    print('temperature scaling')
    test_calibration('temp_scaling', 'temperature_scaling_test10', load_model = False, load_model_path = 'test_calibration_model.pt', load_features = True, label_smoothing = 'none')
    print('histogram binning')
    test_calibration('histogram_binning', 'temperature_scaling_test10', load_model = False, load_model_path = 'test_calibration_model.pt', load_features = True, label_smoothing = 'none')
    print('isotonic regression')
    test_calibration('isotonic_regression', 'temperature_scaling_test10', load_model = False, load_model_path = 'test_calibration_model.pt', load_features = True, label_smoothing = 'none')
    print('beta calibration')
    test_calibration('beta_calibration', 'temperature_scaling_test10', load_model = False, load_model_path = 'test_calibration_model.pt', load_features = True, label_smoothing = 'none')
    print('platt scaling')
    test_calibration('platt_scaling', 'temperature_scaling_test10', load_model = False, load_model_path = 'test_calibration_model.pt', load_features = True, label_smoothing = 'none')
    print('equal_freq_binning')
    test_calibration('equal_freq_binning', 'temperature_scaling_test10', load_model = False, load_model_path = 'test_calibration_model.pt', load_features = True, label_smoothing = 'none')
    print('ensemble_temperature_scaling')
    test_calibration('ensemble_temperature_scaling', 'temperature_scaling_test10', load_model = False, load_model_path = 'test_calibration_model.pt', load_features = True, label_smoothing = 'none')
    print('bbq')
    test_calibration('bbq', 'temperature_scaling_test10', load_model = False, load_model_path = 'test_calibration_model.pt', load_features = True, label_smoothing = 'none')
    print('enir')
    test_calibration('enir', 'temperature_scaling_test10', load_model = False, load_model_path = 'test_calibration_model.pt', load_features = True, label_smoothing = 'none')
    print('platt_binning')
    test_calibration('platt_binning', 'temperature_scaling_test10', load_model = False, load_model_path = 'test_calibration_model.pt', load_features = True, label_smoothing = 'none')
    print('vector_scaling')
    test_calibration('vector_scaling', 'temperature_scaling_test10', load_model=False, load_model_path = None, load_features=True, label_smoothing = 'none')
    print('matrix_scaling')
    test_calibration('matrix_scaling', 'temperature_scaling_test10', load_model=False, load_model_path = None, load_features=True, label_smoothing = 'none')

def main(models, # list of models 
         dataset, # any of keys in dataset_name_to_load_func 
         recalibration_method, # any of keys in methods
         folder_name, # where to save results
         criterion= criterion, 
         load_features = False, # load previously computed features for dataset
         calibrate = True, # include calibration step to adjust confidence scores
         label_smoothing = 'none', # for label smoothing during calibration fitting; 'none', 'alpha', or 'platt'
         label_smoothing_alpha=0.1, # alpha parameter used if label_smoothing = 'alpha'
         retrain_models_from_scratch=False, # if True, start with untrained model in each self-training iteration
         include_all_data_when_retraining=True, # if True, training set in each self-training iteration includes training set in previous iteration plus newly added unlabeled samples
         label_smoothing_model_training=False, # alpha label smoothing during model training
         label_smoothing_model_training_alpha=0.1, 
         labeled_percentage=0.2, # percentage of training data to use as initial set of labeled data (for training)
         validation_percentage=0.1, # percentage of training data to use as validation set for determining calibration parameters
         learning_rate=0.1, # for model training
         balance_classes=False, # if True, ensure initial training set has equal number of instances per class
         margin=False, # see explanation in unlabeled_samples_to_train function
         margin_only=False, # see explanation in unlabeled_samples_to_train function
         k_best=None, # see explanation in unlabeled_samples_to_train function
         k=1000, # see explanation in unlabeled_samples_to_train function
         k_best_and_threshold=None, # see explanation in unlabeled_samples_to_train function
         diff_between_top_two_classes=0.3, # see explanation in unlabeled_samples_to_train function
         validation_model_dataset_separate=True, # if True, separate validation sets for model tuning and calibration fitting
         validation_model_percentage=0.1, # percentage of training data to use as validation set for tuning model
         num_epochs=10,
         batch_size=64):
    '''
    Runs self-training with calibration pipeline
    '''
    # reproducible
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    no_calibration = not calibrate 

    # get featurized dataset
    if load_features: 
        train = combine_dataset_files(dataset, 'train', '.')

        # first check if test set for this dataset exists
        if os.path.exists(dataset + '_1_features_test_data.pt'):
            test = combine_dataset_files(dataset, 'test', '.')
        else:
            test = None

        # first check if unlabeled set for this dataset exists
        if os.path.exists(dataset + '_1_features_unlabeled_data.pt'):
            unlabeled = combine_dataset_files(dataset, 'unlabeled', '.')
        else:
            unlabeled = None 

        if validation_model_dataset_separate:
            validation_model_size = int(validation_model_percentage * len(train[1]))
            train = train[0][:validation_model_size], train[1][:validation_model_size]

        (train_features, train_labels), validation, (test_features, test_labels), (unlabeled_features, unlabeled_labels) = split_datasets(train, labeled_proportion=labeled_percentage, test=test, unlabeled=unlabeled, validation_proportion=validation_percentage, no_calibration=no_calibration, balance_classes=balance_classes)

        if validation is not None:
            validation_features, validation_labels = validation 
            validation_dataloader = DataLoader(TextDataset(validation_features, validation_labels), batch_size=batch_size)

        train_dataset_size = len(train_labels)
        train_dataloader = DataLoader(TextDataset(train_features, train_labels), batch_size=batch_size)
        test_dataloader = DataLoader(TextDataset(test_features, test_labels), batch_size=batch_size)
        unlabeled_dataloader = DataLoader(TextDataset(unlabeled_features, unlabeled_labels), batch_size=batch_size)
    else:
        dataset_name_to_load_func = {'imdb': load_imdb_dataset, 'sst2': load_sst2_dataset, 'sst5': load_sst5_dataset, 'amazon_elec': load_amazon_elec_dataset, 'amazon_elec_binary': load_amazon_elec_binary_dataset, 'dbpedia': load_dbpedia_dataset, 'ag_news': load_ag_news_dataset, 'yelp_full': load_yelp_full_dataset, 'yelp_polarity': load_yelp_polarity_dataset, 'amazon_full': load_amazon_full_dataset, 'amazon_polarity': load_amazon_polarity_dataset, 'yahoo': load_yahoo_answers_dataset, 'twenty_news': load_twenty_news_dataset, 'airport_tweets': load_airport_tweets_dataset, 'modified_amazon_elec_binary': load_modified_amazon_elec_binary_dataset}

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
        train = featurize_dataset(train_dataset, device, batch_size, dataset, 'train_data.pt')
        if test is not None:
            test = featurize_dataset(test_dataset, device, batch_size, dataset, 'test_data.pt')
        if unlabeled is not None:
            unlabeled = featurize_dataset(unlabeled_dataset, device, batch_size, dataset, 'unlabeled_data.pt')

        if validation_model_dataset_separate:
            validation_model_size = int(validation_model_percentage * len(train_labels))
            train = train_features[:validation_model_size], train_labels[:validation_model_size]

        # split datasets to get validation and updated train and unlabeled sets
        (train_features, train_labels), validation, (test_features, test_labels), (unlabeled_features, unlabeled_labels) = split_datasets(train, labeled_proportion=labeled_percentage, test=test, unlabeled=unlabeled, validation_proportion=validation_percentage, no_calibration=no_calibration, balance_classes=balance_classes)

        if validation is not None:
            validation_features, validation_labels = validation 
            validation_dataloader = DataLoader(TextDataset(validation_features, validation_labels), batch_size=batch_size)

        train_dataset_size = len(train_labels)

        # create dataloaders 
        train_dataloader = DataLoader(TextDataset(train_features, train_labels), batch_size=batch_size)
        test_dataloader = DataLoader(TextDataset(test_features, test_labels), batch_size=batch_size)
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

    all_unlabeled_data_used = False # for stopping criteria

    for i in range(num_self_training_iterations):

        # ensure new model for each iteration
        for j in range(len(models)):
            models[j] = model_training(models[j], device, train_dataloader, criterion, num_epochs = num_epochs, label_smoothing=label_smoothing_model_training, label_smoothing_alpha=label_smoothing_model_training_alpha, learning_rate = learning_rate)

        # predictions on test set
        pre_softmax_probs, post_softmax_probs, predicted_probs, predicted_labels, true_labels, _ = get_model_predictions(test_dataloader, device, models, use_pre_softmax=False, use_post_softmax=True)

        # check calibration
        if calibrate:
            ece, _, _, _, _ = plot_calibration_curve(true_labels, post_softmax_probs, folder_name + '/' + dataset + '_' + recalibration_method + '_iteration' + str(i) + '_test_initial_calibration.jpg')
        else:
            ece, _, _, _, _ = plot_calibration_curve(true_labels, post_softmax_probs, folder_name + '/' + dataset + '_no_calibration_iteration' + str(i) + '_test_initial_calibration.jpg')

        # update metrics
        accuracy.append(accuracy_score(true_labels, predicted_labels))
        precision.append(precision_score(true_labels, predicted_labels, average='macro'))
        recall.append(recall_score(true_labels, predicted_labels, average='macro'))
        f1.append(f1_score(true_labels, predicted_labels, average='macro'))
        # auc_roc.append(roc_auc_score(true_labels, predicted_probs))
        training_data_size.append(train_dataset_size)
        expected_calibration_errors.append(ece)

        # recalibrate 
        if calibrate:
            methods = {'histogram_binning': calibrate_histogram_binning, 'isotonic_regression': calibrate_isotonic_regression, 'beta_calibration': calibrate_beta_calibration, 'temp_scaling': calibrate_temperature_scaling, 'platt_scaling': calibrate_platt_scaling, 'equal_freq_binning': calibrate_equal_freq_binning, 'bbq': calibrate_bbq, 'ensemble_temperature_scaling': calibrate_ensemble_temperature_scaling, 'enir': calibrate_enir, 'platt_binning': calibrate_platt_binner, 'vector_scaling': calibrate_vector_scaling, 'matrix_scaling': calibrate_matrix_scaling}
            calibration_class = methods[recalibration_method]

            if recalibration_method in ('temp_scaling', 'platt_scaling', 'ensemble_temperature_scaling'):
                calibrated_probs = calibration_class(validation_dataloader, device, models, pre_softmax_probs, label_smoothing, label_smoothing_alpha)
            elif recalibration_method in ('vector_scaling', 'matrix_scaling'):
                calibrated_probs = calibration_class(validation_dataloader, device, models, pre_softmax_probs, label_smoothing, num_classes=num_classes) # need to chnage param at top of file
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

        # select unlabeled samples to add to training set
        if calibrate:
            train_dataloader, unlabeled_dataloader, num_samples_added_to_train, num_unlabeled_samples = unlabeled_samples_to_train(models, device, unlabeled_dataloader, num_classes=num_classes, calibrate=True, validation_dataloader=validation_dataloader, recalibration_method=recalibration_method, label_smoothing=label_smoothing, label_smoothing_alpha=label_smoothing_alpha, retrain_from_scratch=include_all_data_when_retraining, train_dataloader=train_dataloader, batch_size=64, margin=margin, margin_only=margin_only, k_best_and_threshold=k_best_and_threshold, k_best=k_best, threshold=threshold, k = k, diff_between_top_two_classes=diff_between_top_two_classes)
        else:
            train_dataloader, unlabeled_dataloader, num_samples_added_to_train, num_unlabeled_samples = unlabeled_samples_to_train(models, device, unlabeled_dataloader, num_classes=num_classes, calibrate=False, label_smoothing=label_smoothing, label_smoothing_alpha=label_smoothing_alpha, retrain_from_scratch=include_all_data_when_retraining, train_dataloader=train_dataloader, batch_size=64, margin=margin, margin_only=margin_only, k_best_and_threshold=k_best_and_threshold, k_best=k_best, threshold=threshold, k = k, diff_between_top_two_classes=diff_between_top_two_classes)


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
            for i in range(len(models)):
                old_model = models[i]
                new_model_class = old_model.__class__ 
                new_model = new_model_class(old_model.embed_dim, old_model.num_class, initrange=old_model.initrange) # assumes model of type TextClassificationModel
                models[i] = new_model

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

    if recalibration_method is None:
        recalibration_method = 'no_calibration'
    plt.savefig(folder_name + '/' + dataset + '_' + recalibration_method + '_metrics.jpg')

    plt.figure()
    plt.plot(training_data_size)
    plt.savefig(folder_name + '/' + dataset + '_' + recalibration_method + '_trainingdatasize.jpg')

    # get/store metric values
    metrics_dict = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1, 'ece': expected_calibration_errors, 'training_data_size': training_data_size}
    if calibrate:
        metrics_dict['ece_after_recalibration'] = expected_calibration_errors_recalibration
    metrics = pd.DataFrame(data=metrics_dict)
    metrics.to_csv(folder_name + '/' + dataset + '_' + recalibration_method + '_metrics.csv')

def run_experiments():
    '''
    Running main() with different parameters
    '''
    # vary labeled_percentage, validation_percentage
    adding_unlabeled_samples_methods = ['k_best_and_threshold', 'k_best', 'threshold']
    binary_classification_calibration_methods = ['histogram_binning', 'isotonic_regression', 'beta_calibration', 'temp_scaling', 'platt_scaling', 'equal_freq_binning', 'bbq', 'ensemble_temperature_scaling', 'enir', 'platt_binning']

    initrange = 0.2
    model = TextClassificationModel(768, 2, initrange=initrange)
    models = [model]
    dataset = 'imdb'
    labeled_percentages = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    validation_percentages = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    sum_labeled_validation = list(set([x + y for x, y in zip(labeled_percentages, validation_percentages)]))
    labeled_percentages_no_calibration = list(set(labeled_percentages + sum_labeled_validation))

    # binary_classification_datasets = ['imdb', 'sst2', 'airport_tweets', 'yelp_polarity', 'amazon_elec_binary', 'modified_elec_binary']

    # first without calibration
    for unlabeled_samples_method in adding_unlabeled_samples_methods:
        for retrain_models_from_scratch_value in [True, False]:
            for labeled_percentage in labeled_percentages_no_calibration:
                folder_name = "self_training_imdb_" + unlabeled_samples_method + "_" + str(retrain_models_from_scratch_value) + "_no_calibration_" + str(labeled_percentage)
                folder_exists = os.path.exists(folder_name)
                if not folder_exists:
                    os.makedirs(folder_name)

                    if unlabeled_samples_method == 'k_best_and_threshold':
                        main(models, dataset, criterion, None, folder_name, load_features=True, calibrate=False, retrain_models_from_scratch=retrain_models_from_scratch_value, k_best=False, k_best_and_threshold=True, labeled_percentage=labeled_percentage)
                    elif unlabeled_samples_method == 'k_best':
                        main(models, dataset, criterion, None, folder_name, load_features=True, calibrate=False, retrain_models_from_scratch=retrain_models_from_scratch_value, k_best=True, k_best_and_threshold=False, labeled_percentage=labeled_percentage)  
                    else:
                        main(models, dataset, criterion, None, folder_name, load_features=True, calibrate=False, retrain_models_from_scratch=retrain_models_from_scratch_value, k_best=False, k_best_and_threshold=False, labeled_percentage=labeled_percentage)             

    # second with all binary classification calibration methods 
    for unlabeled_samples_method in adding_unlabeled_samples_methods:
        for retrain_models_from_scratch_value in [True, False]:
            for calibration_method in binary_classification_calibration_methods:
                for labeled_percentage in labeled_percentages:
                    for validation_percentage in validation_percentages:
                        folder_name = "self_training_imdb_" + unlabeled_samples_method + "_" + str(retrain_models_from_scratch_value) + "_" + calibration_method + "_" + str(labeled_percentage) + "_" + str(validation_percentage) 
                        folder_exists = os.path.exists(folder_name)
                        if not folder_exists:
                            os.makedirs(folder_name)
                        
                            if unlabeled_samples_method == 'k_best_and_threshold':
                                main(models, dataset, criterion, calibration_method, folder_name, load_features=True, calibrate=True, retrain_models_from_scratch=retrain_models_from_scratch_value, k_best=False, k_best_and_threshold=True, labeled_percentage=labeled_percentage, validation_percentage=validation_percentage)
                            elif unlabeled_samples_method == 'k_best':
                                main(models, dataset, criterion, calibration_method, folder_name, load_features=True, calibrate=True, retrain_models_from_scratch=retrain_models_from_scratch_value, k_best=True, k_best_and_threshold=False, labeled_percentage=labeled_percentage, validation_percentage=validation_percentage)  
                            else:
                                main(models, dataset, criterion, calibration_method, folder_name, load_features=True, calibrate=True, retrain_models_from_scratch=retrain_models_from_scratch_value, k_best=False, k_best_and_threshold=False, labeled_percentage=labeled_percentage, validation_percentage=validation_percentage)   

