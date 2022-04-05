import torch 
import torch.optim as optim
from torch import nn
import numpy as np
import calibration

def model_training(model, device, num_epochs, train_loader, criterion, file_name = None, label_smoothing=False, label_smoothing_alpha=0.1, learning_rate=0.001):
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate) # add changing optimizer + parameters used 

    model.train()

    for _ in range(num_epochs):

        for (_, batch) in enumerate(train_loader):
            inputs, labels = batch['Text'].to(device), batch['Class'].to(device)

            if label_smoothing:
                labels = calibration.alpha_label_smoothing(labels, label_smoothing_alpha)
            
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            labels = torch.flatten(labels)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    if file_name is not None:
        torch.save(model.state_dict(), file_name)

    return model

def get_model_predictions(dataloader, device, models, use_pre_softmax=False, use_post_softmax=False, unlabeled=False):
    all_pre_softmax_probs = []
    all_post_softmax_probs = []
    all_predicted_probs = []
    true_labels = None
    texts = None

    for model in models:
        pre_softmax_probs = []
        post_softmax_probs = []
        predicted_probs = []
        get_true_labels = False
        get_texts = False

        if true_labels is None and not unlabeled: # need to get true_labels once (same for all models)
            true_labels = []
            get_true_labels = True
        if texts is None: 
            texts = []
            get_texts = True

        model.eval()

        with torch.no_grad():
            for (_, batch) in enumerate(dataloader):
                if unlabeled:
                    inputs = batch['Text'].to(device)
                else:
                    inputs, labels = batch['Text'].to(device), batch['Class'].to(device)

                pre_softmax = model(inputs)

                sm = nn.Softmax(dim=1)
                post_softmax = sm(pre_softmax)

                predicted_prob, _ = torch.max(post_softmax.data, 1)

                pre_softmax_probs.append(pre_softmax)
                post_softmax_probs.append(post_softmax)
                predicted_probs.extend(predicted_prob.cpu().numpy())
                
                if get_true_labels:
                    true_labels.extend(labels.cpu().numpy())
                if get_texts:
                    texts.append(inputs)

    
        pre_softmax_probs = torch.cat(pre_softmax_probs)
        post_softmax_probs = torch.cat(post_softmax_probs)
        predicted_probs = torch.tensor(np.array(predicted_probs).reshape(-1, 1))

        if get_true_labels:
            true_labels = np.array(true_labels)
        if get_texts:
            texts = torch.cat(texts).cpu().numpy()

        all_pre_softmax_probs.append(pre_softmax_probs)
        all_post_softmax_probs.append(post_softmax_probs)
        all_predicted_probs.append(predicted_probs)
    
    # average pre_softmax, post_softmax, predicted probs
    aggregate_pre_softmax_probs = torch.mean(torch.stack(all_pre_softmax_probs), dim=0)
    sm = nn.Softmax(dim=1)
    aggregate_post_softmax_probs_from_pre = sm(aggregate_pre_softmax_probs)
    aggregate_predicted_probs_from_pre, aggregate_predicted_labels_from_pre = torch.max(aggregate_post_softmax_probs_from_pre.data, 1)

    aggregate_post_softmax_probs = torch.mean(torch.stack(all_post_softmax_probs), dim=0)
    aggregate_predicted_probs_from_post, aggregate_predicted_labels_from_post = torch.max(aggregate_post_softmax_probs.data, 1)

    # get predicted_labels 
    aggregate_predicted_probs = torch.mean(torch.stack(all_predicted_probs), dim=0)
    _, aggregate_predicted_labels_from_predicted = torch.max(aggregate_predicted_probs.data, 1)

    if use_pre_softmax:
        return aggregate_pre_softmax_probs.cpu().numpy(), aggregate_post_softmax_probs_from_pre.cpu().numpy(), aggregate_predicted_probs_from_pre.cpu().numpy(), aggregate_predicted_labels_from_pre.cpu().numpy(), true_labels, texts
    elif use_post_softmax:
        return aggregate_pre_softmax_probs.cpu().numpy(), aggregate_post_softmax_probs.cpu().numpy(), aggregate_predicted_probs_from_post.cpu().numpy(), aggregate_predicted_labels_from_post.cpu().numpy(), true_labels, texts
    else:
        return aggregate_pre_softmax_probs.cpu().numpy(), aggregate_post_softmax_probs.cpu().numpy(), aggregate_predicted_probs.cpu().numpy(), aggregate_predicted_labels_from_predicted.cpu().numpy(), true_labels, texts

