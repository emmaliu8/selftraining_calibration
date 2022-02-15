import torch 
import torch.optim as optim
from torch import nn
import numpy as np

def model_training(model, device, num_epochs, train_loader, criterion, file_name = None):
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001) # add changing optimizer + parameters used 

    model.train()

    for _ in range(num_epochs):

        for (_, batch) in enumerate(train_loader):
            inputs, labels = batch['Text'].to(device), batch['Class'].to(device)
            
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

def get_model_predictions(dataloader, device, model):
    pre_softmax_probs = []
    post_softmax_probs = []
    predicted_probs = []
    predicted_labels = [] 
    true_labels = []

    model.eval()

    with torch.no_grad():
        for (_, batch) in enumerate(dataloader):
            inputs, labels = batch['Text'].to(device), batch['Class'].to(device)

            pre_softmax = model(inputs) 

            # Get softmax values for net input and resulting class predictions
            sm = nn.Softmax(dim=1)
            post_softmax = sm(pre_softmax)

            predicted_prob, predicted_label = torch.max(post_softmax.data, 1)

            pre_softmax_probs.append(pre_softmax)
            post_softmax_probs.append(post_softmax)
            predicted_probs.extend(predicted_prob.cpu().numpy())
            predicted_labels.extend(predicted_label.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    pre_softmax_probs = torch.cat(pre_softmax_probs).cpu().numpy()
    post_softmax_probs = torch.cat(post_softmax_probs).cpu().numpy()
    predicted_probs = np.array(predicted_probs).reshape(-1, 1)
    predicted_labels = np.array(predicted_labels)
    true_labels = np.array(true_labels)

    return pre_softmax_probs, post_softmax_probs, predicted_probs, predicted_labels, true_labels
    