import torch 
from torch import nn
import numpy as np 
from data_setup import create_dataset, split_datasets, load_sst5_dataset, TextDataset
from torch.utils.data import DataLoader
from extract_features import featurize_dataset
from calibration import calibrate_platt_scaling, plot_calibration_curve, calibrate_temperature_scaling, calibrate_histogram_binning, calibrate_isotonic_regression, calibrate_beta_calibration, calibrate_equal_freq_binning, calibrate_bbq, calibrate_ensemble_temperature_scaling, calibrate_enir, calibrate_platt_binner, calibrate_vector_scaling, calibrate_matrix_scaling
from classifiers import TextClassificationModel
from model_training import model_training, get_model_predictions
from sklearn.metrics import accuracy_score

# CONSTANTS
batch_size = 64
threshold = 0.8 # used to determine which unlabeled examples have high enough confidence
num_classes = 2 
num_epochs = 10
learning_rate = 0.1
num_self_training_iterations = 1000000

criterion = nn.CrossEntropyLoss()

def test_calibration_multiclass(calibration_method, folder_name, load_model = False, load_model_path = None, load_features = False, label_smoothing = 'none'):
    '''
    Testing calibration methods on a multi-class dataset (SST-5)

    folder_name: where to store results
    load_model: True if loading previously trained model, False if training from scratch
    load_features: True if loading previously saved features, False if featurizing from scratch
    '''
    # reproducible
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if load_features:
        train_features = torch.load('sst5_features_train_data.pt')
        train_labels = torch.load('sst5_labels_train_data.pt')

        test_features = torch.load('sst5_features_test_data.pt')
        test_labels = torch.load('sst5_labels_test_data.pt')

        (train_features, train_labels), (validation_features, validation_labels), (test_features, test_labels), (unlabeled_features, unlabeled_labels) = split_datasets((train_features, train_labels), test=(test_features, test_labels), labeled_proportion=0.1, validation_proportion=0.1)

        train_dataloader = DataLoader(TextDataset(train_features, train_labels), batch_size=batch_size)
        test_dataloader = DataLoader(TextDataset(test_features, test_labels), batch_size=batch_size)
        validation_dataloader = DataLoader(TextDataset(validation_features, validation_labels), batch_size=batch_size)
    else:
        train, unlabeled, test = load_sst5_dataset('../')

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
        (train_features, train_labels), (validation_features, validation_labels), (test_features, test_labels), (unlabeled_features, unlabeled_labels) = split_datasets(train, test=test, unlabeled=unlabeled, labeled_proportion=0.1, validation_proportion=0.9)

        train_dataloader = DataLoader(TextDataset(train_features, train_labels), batch_size=batch_size)
        test_dataloader = DataLoader(TextDataset(test_features, test_labels), batch_size=batch_size)
        validation_dataloader = DataLoader(TextDataset(validation_features, validation_labels), batch_size=batch_size)

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
        calibrated_train_probs = calibration_class(validation_dataloader, device, [model], train_pre_softmax_probs, label_smoothing, num_classes=5) # testing on sst5
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
        calibrated_test_probs = calibration_class(validation_dataloader, device, [model], test_pre_softmax_probs, label_smoothing, num_classes=5) # testing on sst5
    else:
        calibrated_test_probs = calibration_class(validation_dataloader, device, [model], test_post_softmax_probs, label_smoothing)

    plot_calibration_curve(test_true_labels, calibrated_test_probs, folder_name + '/' + calibration_method + '_test_after_calibration.jpg')

# print('matrix scaling test')
# test_calibration_multiclass('matrix_scaling', 'temperature_scaling_test10', load_model=False, load_model_path = None, load_features=True, label_smoothing = 'none')
# print('vector scaling test')
# test_calibration_multiclass('vector_scaling', 'temperature_scaling_test10', load_model=False, load_model_path = None, load_features=True, label_smoothing = 'none')
print('full dirichlet test')
test_calibration_multiclass('vector_scaling', 'temperature_scaling_test10', load_model=False, load_model_path = None, load_features=True, label_smoothing = 'none')