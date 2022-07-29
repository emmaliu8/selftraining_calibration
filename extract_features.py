import torch
import transformers
import numpy as np
from torch.utils.data import DataLoader
import os

batch_size = 256

def remove_br_tags(text):
    return text.replace("<br /><br />", " ")

def tokenize_and_features(text, device):
    '''
    Convert text (list of strings) into features
    '''
    # assumes text is a list of samples
    text = [remove_br_tags(element) for element in text]
    bert_tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')
    bert_model = transformers.BertModel.from_pretrained('bert-base-uncased')
    bert_model = bert_model.to(device)

    input_ids = []
    attention_masks = []
    for example in text: 
        tokenized = bert_tokenizer.encode_plus(example, add_special_tokens=True, return_attention_mask=True, max_length=512, truncation=True, pad_to_max_length=True)
        input_ids.append(torch.tensor(tokenized['input_ids']))
        attention_masks.append(torch.tensor(tokenized['attention_mask']))
    input_ids = torch.transpose(torch.stack(input_ids, dim=1), 0, 1).to(device)
    attention_masks = torch.transpose(torch.stack(attention_masks, dim=1), 0, 1).to(device)
    with torch.no_grad():
        outputs = bert_model(input_ids, attention_mask=attention_masks)
    features = outputs[0][:, 0, :].cpu().numpy()
    return features

def featurize_dataset(dataset, device, batch_size, dataset_name = None, file_name = None, start_index=0):
    '''
    Create version of dataset with text converted to features
    Saves dataset to files in batches of 100000

    start_index used if part of the dataset has already been featurized, should be set to the index of the last saved file
    '''
    text_dataloader = DataLoader(dataset, batch_size=batch_size)
    features = None
    all_features = None
    labels = []
    index = 1
    added = False

    for (_, batch) in enumerate(text_dataloader):
        batch_labels = batch['Class']
        labels.extend(batch_labels)

        if index > start_index:
            new_features = tokenize_and_features(batch['Text'], device)

            if features is None:
                features = new_features
            else:
                features = np.concatenate((features, new_features), axis=0)
            
            if all_features is None:
                all_features = new_features 
            else:
                all_features = np.concatenate((all_features, new_features), axis=0)
        else:
            if not added: 
                # loads previously featurized samples from dataset
                # assumes loaded features were created with featurize_dataset
                new_features = torch.load(dataset_name + '_' + str(index) + '_features_' + file_name)
                features = new_features

                if all_features is None:
                    all_features = new_features
                else:
                    all_features = np.concatenate((all_features, new_features), axis=0)
                added = True

        if len(labels) > 100000:
            if file_name is not None:
                torch.save(features, dataset_name + '_' + str(index) + '_features_' + file_name)
                torch.save(labels, dataset_name + '_' + str(index) + '_labels_' + file_name)
                features = None 
                labels = []
                index += 1
                added = False
    
    # save dataset to file
    if file_name is not None:
        torch.save(features, dataset_name + '_' + str(index) + '_features_' + file_name)
        torch.save(labels, dataset_name + '_' + str(index) + '_labels_' + file_name)

    return all_features, dataset.get_labels()

def combine_dataset_files(dataset, split, file_path):
    '''
    Combines dataset that has been featurized and saved to files (from featurize_dataset)

    split: 'train', 'test', 'validation', or 'unlabeled'
    '''
    # format of features filename is {dataset}_{index}_features_{split}_data.pt
    # format of labels filename is {dataset}_{index}_labels_{split}_data.pt

    largest_index = 1
    for fname in sorted(os.listdir(file_path)):
        if fname.startswith(dataset) and fname.endswith(split + "_data.pt"):
            fname_split = fname.split("_")
            index_of_num = None
            for idx in range(len(fname_split)):
                if fname_split[idx][0] in '1234567890':
                    index_of_num = idx
                    break
            if int(fname_split[index_of_num]) > largest_index:
                largest_index = int(fname_split[index_of_num])
    
    features = [] 
    labels = []
    for i in range(1, largest_index + 1):
        new_features = torch.load(dataset + "_" + str(i) + "_features_" + split + "_data.pt")
        new_labels = torch.load(dataset + "_" + str(i) + "_labels_" + split + "_data.pt")

        features.extend(new_features)
        labels.extend(new_labels)

    return features, labels
