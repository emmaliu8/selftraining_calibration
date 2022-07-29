import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from data_setup import TextDataset, get_dataset_from_dataloader
from model_training import get_model_predictions
from calibration import calibrate_platt_scaling, calibrate_temperature_scaling, calibrate_histogram_binning, calibrate_isotonic_regression, calibrate_beta_calibration, calibrate_equal_freq_binning, calibrate_bbq, calibrate_ensemble_temperature_scaling, calibrate_enir, calibrate_platt_binner, calibrate_vector_scaling, calibrate_matrix_scaling

def unlabeled_samples_to_train(models, 
                               device, 
                               unlabeled_dataloader,
                               num_classes = 2,
                               calibrate = False, 
                               validation_dataloader=None, # needed if calibrate = True, for fitting calibration method
                               recalibration_method=None, # needed if calibrate = True
                               label_smoothing = 'none', # during fitting of calibration method
                               label_smoothing_alpha = 0.5, # used if label_smoothing = 'alpha'
                               retrain_from_scratch = False,
                               train_dataloader=None, # needed if retrain_from_scratch = True
                               batch_size = 64, 
                               margin = False, # consider margin for any method for picking unlabeled samples, must be True if margin_only is True
                               margin_only = False,
                               k_best_and_threshold = None,
                               k_best = None,
                               threshold = 0.8, 
                               k = 1000, 
                               diff_between_top_two_classes = 0.3, # parameter for margin method
                            ):
    '''
    Selects unlabeled samples on which the model(s) have high confidence to use as training samples in the next iteration of self-training

    If calibrate = True: uses calibrated probabilities as the confidence scores

    If retrain_from_scratch = True: Training set in next iteration of self-training includes newly added unlabeled samples in addition to the training set from the previous iteration of self-training

    Allowed methods for selecting unlabeled samples:
    - Margin: margin = True, margin_only = True
    - K-Best and Threshold with Margin: margin = True, k_best_and_threshold = True, margin_only = False
    - K-Best and Threshold without Margin: margin = False, k_best_and_threshold = True, margin_only = False
    - K-Best with Margin: margin = True, k_best = True, margin_only = False, k_best_and_threshold = False
    - K-Best without Margin: margin = False, k_best = True, margin_only = False, k_best_and_threshold = False
    - Threshold with Margin: margin = True, k_best = False, margin_only = False, k_best_and_threshold = False
    - Threshold without Margin: margin = False, k_best = False, margin_only = False, k_best_and_threshold = False

    Margin: Difference between probabilities for the top two classes must be at least the given margin
    K-Best: Select samples with the k-highest confidence scores
    Threshold: Selected samples must have confidence higher than given threshold
    '''

    # get predictions on unlabeled set
    aggregate_pre_softmax_probs, aggregate_post_softmax_probs, aggregate_predicted_probs, aggregate_predicted_labels, _, all_unlabeled_inputs = get_model_predictions(unlabeled_dataloader, device, models, use_pre_softmax=False, use_post_softmax=True, unlabeled=True)

    # modify probs for unlabeled if calibration enabled
    if calibrate:
        methods = {'histogram_binning': calibrate_histogram_binning, 'isotonic_regression': calibrate_isotonic_regression, 'beta_calibration': calibrate_beta_calibration, 'temp_scaling': calibrate_temperature_scaling, 'platt_scaling': calibrate_platt_scaling, 'equal_freq_binning': calibrate_equal_freq_binning, 'bbq': calibrate_bbq, 'ensemble_temperature_scaling': calibrate_ensemble_temperature_scaling, 'enir': calibrate_enir, 'platt_binning': calibrate_platt_binner, 'vector_scaling': calibrate_vector_scaling, 'matrix_scaling': calibrate_matrix_scaling}
        calibration_class = methods[recalibration_method]

        if recalibration_method in ('temp_scaling', 'platt_scaling', 'ensemble_temperature_scaling', 'platt_binning'):
            unlabeled_calibrated_probs = calibration_class(validation_dataloader, device, models, aggregate_pre_softmax_probs, label_smoothing = label_smoothing, label_smoothing_alpha = label_smoothing_alpha)
        elif recalibration_method in ('vector_scaling', 'matrix_scaling'):
            unlabeled_calibrated_probs = calibration_class(validation_dataloader, device, models, aggregate_pre_softmax_probs, label_smoothing = label_smoothing, label_smoothing_alpha = label_smoothing_alpha, num_classes=num_classes)
        else:
            unlabeled_calibrated_probs = calibration_class(validation_dataloader, device, models, aggregate_post_softmax_probs, label_smoothing = label_smoothing, label_smoothing_alpha = label_smoothing_alpha)
    
    # get all unlabeled inputs
    all_unlabeled_inputs = pd.DataFrame(all_unlabeled_inputs)

    df = pd.DataFrame([])

    if calibrate:
        df['predicted_probs'] = np.max(unlabeled_calibrated_probs, axis=1)   
        df['predicted_labels'] = np.argmax(unlabeled_calibrated_probs, axis=1)
    else:
        df['predicted_probs'] = np.squeeze(aggregate_predicted_probs)
        df['predicted_labels'] = aggregate_predicted_labels
    
    # get margin for each sample
    if margin:
        if calibrate:
            sorted_unlabeled_calibrated_probs = np.sort(unlabeled_calibrated_probs, axis=1) # sorted in ascending order 
            margin_values = sorted_unlabeled_calibrated_probs[:, -1] - sorted_unlabeled_calibrated_probs[:, -2]
            df['margin'] = margin_values            
            df['margin'] = df['margin'].abs()
        else:
            sorted_unlabeled_probs = np.sort(aggregate_post_softmax_probs, axis=1) # sorted in ascending order 
            margin_values = sorted_unlabeled_probs[:, -1] - sorted_unlabeled_probs[:, -2]
            df['margin'] = margin_values
            df['margin'] = df['margin'].abs()

    # apply method for choosing unlabeled samples
    if margin_only:
        high_margin = df.loc[df['margin'] > diff_between_top_two_classes]
        temp = np.array(all_unlabeled_inputs.drop(index=high_margin.index)) # samples to unlabeled set

        unlabeled_texts = temp
        temp2 = np.array(all_unlabeled_inputs.loc[high_margin.index]) # samples to training set

        unlabeled_to_train_texts = temp2
        temp3 = df['predicted_labels'].loc[high_margin.index] # labels for samples to training set

        unlabeled_to_train_labels = temp3
    elif k_best_and_threshold:
        df_sorted = df.sort_values(by=['predicted_probs'], ascending=False)
        df_k_best = df_sorted.head(k)
        if margin:
            df_k_best_and_threshold = df_k_best.loc[(df_k_best['predicted_probs'] > threshold) & (df_k_best['margin'] > diff_between_top_two_classes)]
        else:
            df_k_best_and_threshold = df_k_best.loc[df_k_best['predicted_probs'] > threshold]
        temp = np.array(all_unlabeled_inputs.drop(index=df_k_best_and_threshold.index))

        unlabeled_texts = temp
        temp2 = np.array(all_unlabeled_inputs.loc[df_k_best_and_threshold.index])

        unlabeled_to_train_texts = temp2 
        temp3 = df['predicted_labels'].loc[df_k_best_and_threshold.index]

        unlabeled_to_train_labels = temp3
    elif k_best:
        df_sorted = df.sort_values(by=['predicted_probs'], ascending=False)
        df_k_best = df_sorted.head(k)
        if margin:
            df_k_best = df_k_best.loc[df_k_best['margin'] > diff_between_top_two_classes]
        temp = np.array(all_unlabeled_inputs.drop(index=df_k_best.index))

        unlabeled_texts = temp
        temp2 = np.array(all_unlabeled_inputs.loc[df_k_best.index])

        unlabeled_to_train_texts = temp2 
        temp3 = df['predicted_labels'].loc[df_k_best.index]

        unlabeled_to_train_labels = temp3
    else:
        if margin:
            high_prob = df.loc[(df['predicted_probs'] > threshold) & (df['margin'] > diff_between_top_two_classes)] 
        else:
            high_prob = df.loc[df['predicted_probs'] > threshold]
        temp = np.array(all_unlabeled_inputs.drop(index=high_prob.index)) # samples to unlabeled set

        unlabeled_texts = temp
        temp2 = np.array(all_unlabeled_inputs.loc[high_prob.index]) # samples to training set

        unlabeled_to_train_texts = temp2
        temp3 = df['predicted_labels'].loc[high_prob.index] # labels for samples to training set

        unlabeled_to_train_labels = temp3

    # construct new train and unlabeled dataloaders
    num_samples_added_to_train = len(unlabeled_to_train_labels)
    unlabeled_to_train_labels = np.array(unlabeled_to_train_labels)
    num_unlabeled_samples_before = unlabeled_texts.shape[0] + num_samples_added_to_train

    unlabeled_dataloader = DataLoader(TextDataset(unlabeled_texts, np.full((unlabeled_texts.shape[0], 1), -1)), batch_size=batch_size)

    if retrain_from_scratch:
        train_text, train_labels = get_dataset_from_dataloader(train_dataloader, device)
        train_text = np.concatenate((train_text, unlabeled_to_train_texts))
        train_labels.extend(unlabeled_to_train_labels)
        train_dataloader = DataLoader(TextDataset(train_text, train_labels), batch_size=batch_size)
    else:
        # train_dataloader should only contain new samples
        train_dataloader = DataLoader(TextDataset(unlabeled_to_train_texts, unlabeled_to_train_labels), batch_size=batch_size)

    return train_dataloader, unlabeled_dataloader, num_samples_added_to_train, num_unlabeled_samples_before
    

