import torch
from torch import nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from data_setup import TextDataset, get_dataset_from_dataloader
from model_training import get_aggregate_model_predictions


def unlabeled_samples_to_train(model, device, train_dataloader, validation_dataloader, unlabeled_dataloader, calibrate, recalibration_method, calibration_class, label_smoothing, threshold, batch_size, retrain_models_from_scratch=True):
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
                if recalibration_method in ('temp_scaling', 'platt_scaling', 'ensemble_temperature_scaling', 'platt_binning'):
                    unlabeled_calibrated_probs = calibration_class(validation_dataloader, device, model, pre_softmax.cpu().numpy(), label_smoothing)

                else:
                    unlabeled_calibrated_probs = calibration_class(validation_dataloader, device, model, post_softmax.cpu().numpy(), label_smoothing)

                unlabeled_calibrated_probs = np.max(unlabeled_calibrated_probs, axis=1)
            
            unlabeled_inputs = pd.DataFrame(np.array(inputs.cpu()))

            df = pd.DataFrame([])

            if calibrate:
                df['predicted_probs'] = unlabeled_calibrated_probs
                df['predicted_labels'] = np.argmax(unlabeled_calibrated_probs, axis=1)
            else:
                df['predicted_probs'] = predicted_prob.cpu()
                df['predicted_labels'] = predicted_label.cpu()

            high_prob = df.loc[df['predicted_probs'] > threshold]
            temp = np.array(unlabeled_inputs.drop(index=high_prob.index)) # samples to unlabeled set

            unlabeled_texts.append(temp)
            temp2 = np.array(unlabeled_inputs.loc[high_prob.index]) # samples to training set

            unlabeled_to_train_texts.append(temp2)
            temp3 = df['predicted_labels'].loc[high_prob.index] # labels for samples to training set

            unlabeled_to_train_labels.extend(df['predicted_labels'].loc[high_prob.index].tolist())


    num_samples_added_to_train = len(unlabeled_to_train_labels)
    unlabeled_to_train_labels = np.array(unlabeled_to_train_labels)
    unlabeled_to_train_texts = np.concatenate(unlabeled_to_train_texts)
    unlabeled_texts = np.concatenate(unlabeled_texts)
    num_unlabeled_samples_before = unlabeled_texts.shape[0] + num_samples_added_to_train

    unlabeled_dataloader = DataLoader(TextDataset(unlabeled_texts, np.full((unlabeled_texts.shape[0], 1), -1)), batch_size=batch_size)

    if retrain_models_from_scratch:
        train_text, train_labels = get_dataset_from_dataloader(train_dataloader, device)
        train_text = np.concatenate((train_text, unlabeled_to_train_texts))
        train_labels.extend(unlabeled_to_train_labels)
        train_dataloader = DataLoader(TextDataset(train_text, train_labels), batch_size=batch_size)
    else:
        train_dataloader = DataLoader(TextDataset(unlabeled_to_train_texts, unlabeled_to_train_labels), batch_size=batch_size)

    return train_dataloader, unlabeled_dataloader, num_samples_added_to_train, num_unlabeled_samples_before


def unlabeled_samples_to_train_multiple_models(models, device, train_dataloader, validation_dataloader, unlabeled_dataloader, calibrate, recalibration_method, calibration_class, label_smoothing, threshold, batch_size):
    aggregate_pre_softmax_probs, aggregate_post_softmax_probs, aggregate_predicted_probs, aggregate_predicted_labels, _, all_unlabeled_inputs = get_aggregate_model_predictions(unlabeled_dataloader, device, models, use_pre_softmax=False, use_post_softmax=True, unlabeled=True)

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
        unlabeled_calibrated_probs = np.max(unlabeled_calibrated_probs, axis=1)
        df['predicted_probs'] = unlabeled_calibrated_probs   
    else:
        df['predicted_probs'] = np.squeeze(aggregate_predicted_probs)
        df['predicted_labels'] = aggregate_predicted_labels

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

    train_text, train_labels = get_dataset_from_dataloader(train_dataloader, device)
    train_text = np.concatenate((train_text, unlabeled_to_train_texts))
    train_labels.extend(unlabeled_to_train_labels)
    train_dataloader = DataLoader(TextDataset(train_text, train_labels), batch_size=batch_size)

    return train_dataloader, unlabeled_dataloader, num_samples_added_to_train, num_unlabeled_samples_before
    

