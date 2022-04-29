import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from data_setup import TextDataset, get_dataset_from_dataloader
from model_training import get_model_predictions


def unlabeled_samples_to_train(models, device, train_dataloader, unlabeled_dataloader, calibrate, recalibration_method, calibration_class, label_smoothing, threshold, batch_size, k_best=None, k=None, k_best_and_threshold=None, margin_only=None, margin=False, diff_between_top_two_classes=None, balance_classes=False, validation_dataloader=None, retrain_from_scratch=True):
    aggregate_pre_softmax_probs, aggregate_post_softmax_probs, aggregate_predicted_probs, aggregate_predicted_labels, _, all_unlabeled_inputs = get_model_predictions(unlabeled_dataloader, device, models, use_pre_softmax=False, use_post_softmax=True, unlabeled=True)

    if calibrate:
        if recalibration_method in ('temp_scaling', 'platt_scaling', 'ensemble_temperature_scaling', 'platt_binning'):
            unlabeled_calibrated_probs = calibration_class(validation_dataloader, device, models, aggregate_pre_softmax_probs, label_smoothing)

        else:
            unlabeled_calibrated_probs = calibration_class(validation_dataloader, device, models, aggregate_post_softmax_probs, label_smoothing)
    
    # get all unlabeled inputs
    all_unlabeled_inputs = pd.DataFrame(all_unlabeled_inputs)

    df = pd.DataFrame([])

    if calibrate:
        df['predicted_labels'] = np.argmax(unlabeled_calibrated_probs, axis=1)
        unlabeled_calibrated_probs_max = np.max(unlabeled_calibrated_probs, axis=1)
        df['predicted_probs'] = unlabeled_calibrated_probs_max   
    else:
        df['predicted_probs'] = np.squeeze(aggregate_predicted_probs)
        df['predicted_labels'] = aggregate_predicted_labels
    
    if margin:
        if calibrate:
            sorted_unlabeled_calibrated_probs = np.sort(unlabeled_calibrated_probs, axis=1) # sorted in ascending order 
            margin = sorted_unlabeled_calibrated_probs[:, -1] - sorted_unlabeled_calibrated_probs[:, -2]
            df['margin'] = margin
        else:
            sorted_unlabeled_probs = np.sort(aggregate_post_softmax_probs, axis=1) # sorted in ascending order 
            margin = sorted_unlabeled_probs[:, -1] - sorted_unlabeled_calibrated_probs[:, -2]
            df['margin'] = margin
            df['margin'] = df['margin'].abs()

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
    

