import torch 
import torch.optim as optim
from torch import nn
import numpy as np
import calibration

from data_setup import get_dataset_from_dataloader

def model_training(model, 
                   device, 
                   train_loader,
                   criterion,
                   num_epochs=10, 
                   file_name = None, 
                   label_smoothing=False, # can only be alpha method so either True or False
                   label_smoothing_alpha=0.1, 
                   learning_rate=0.001):
    '''
    Trains a model on the given dataset (train_loader)

    If file_name is given, saves the trained model
    '''
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate) 

    model.train()

    for _ in range(num_epochs):

        for (_, batch) in enumerate(train_loader):
            inputs, labels = batch['Text'].to(device), batch['Class'].to(device)

            if label_smoothing:
                labels = torch.tensor(calibration.alpha_label_smoothing(labels.cpu(), label_smoothing_alpha, for_model_training=True)).to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            if not label_smoothing:
                labels = torch.flatten(labels)
            else:
                labels = labels.to(torch.float32)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    if file_name is not None:
        torch.save(model.state_dict(), file_name)

    return model

def get_model_predictions(dataloader, 
                          device, 
                          models, # list of models 
                          use_pre_softmax=False, 
                          use_post_softmax=True, # default bc best option for ensembles and does not matter for when there is only one model
                          ):
    '''
    Get predictions from the given model(s) on a dataset
    If the number of models is greater than 1, output is an aggregate of predictions from all inputted models

    Method of aggregation determined by use_pre_softmax and use_post_softmax
    If use_pre_softmax = True: Aggregates predictions before the softmax layer and uses aggregated predictions to get post softmax predictions and predicted labels
    Elif use_post_softmax = True: Aggregates predictions after the softmax layer and uses aggregated predictions to get predicted labels
    '''
    all_pre_softmax_probs = []
    all_post_softmax_probs = []

    # first get texts and true labels
    texts, true_labels = get_dataset_from_dataloader(dataloader, device)
    true_labels = np.array(true_labels)

    for model in models:
        pre_softmax_probs = []
        post_softmax_probs = []

        model.eval()

        with torch.no_grad():
            for (_, batch) in enumerate(dataloader):
                inputs = batch['Text'].to(device)

                pre_softmax = model(inputs)

                sm = nn.Softmax(dim=1)
                post_softmax = sm(pre_softmax)

                pre_softmax_probs.append(pre_softmax)
                post_softmax_probs.append(post_softmax)

        pre_softmax_probs = torch.cat(pre_softmax_probs)
        post_softmax_probs = torch.cat(post_softmax_probs)

        all_pre_softmax_probs.append(pre_softmax_probs)
        all_post_softmax_probs.append(post_softmax_probs)
    
    # average pre_softmax, post_softmax, predicted probs
    aggregate_pre_softmax_probs = torch.mean(torch.stack(all_pre_softmax_probs), dim=0)
    sm = nn.Softmax(dim=1)
    aggregate_post_softmax_probs_from_pre = sm(aggregate_pre_softmax_probs)
    aggregate_predicted_probs_from_pre, aggregate_predicted_labels_from_pre = torch.max(aggregate_post_softmax_probs_from_pre.data, 1)

    aggregate_post_softmax_probs = torch.mean(torch.stack(all_post_softmax_probs), dim=0)
    aggregate_predicted_probs_from_post, aggregate_predicted_labels_from_post = torch.max(aggregate_post_softmax_probs.data, 1)
    aggregate_predicted_probs_from_post = aggregate_predicted_probs_from_post.reshape(-1, 1)

    if use_pre_softmax:
        return aggregate_pre_softmax_probs.cpu().numpy(), aggregate_post_softmax_probs_from_pre.cpu().numpy(), aggregate_predicted_probs_from_pre.cpu().numpy(), aggregate_predicted_labels_from_pre.cpu().numpy(), true_labels, texts
    elif use_post_softmax:
        return aggregate_pre_softmax_probs.cpu().numpy(), aggregate_post_softmax_probs.cpu().numpy(), aggregate_predicted_probs_from_post.cpu().numpy(), aggregate_predicted_labels_from_post.cpu().numpy(), true_labels, texts

