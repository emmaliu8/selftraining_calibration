import torch
import transformers
import numpy as np
from torch.utils.data import DataLoader


batch_size = 256

def remove_br_tags(text):
    return text.replace("<br /><br />", " ")

def tokenize_and_features(text, device):
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

def featurize_dataset(dataset, device, batch_size, dataset_name = None, file_name = None):
    text_dataloader = DataLoader(dataset, batch_size=batch_size)
    features = None
    all_features = None
    labels = []
    index = 1
    added = False

    for (_, batch) in enumerate(text_dataloader):
        batch_labels = batch['Class']
        labels.extend(batch_labels)

        if index > 0:
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