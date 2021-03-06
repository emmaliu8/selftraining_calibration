from torch.utils.data import Dataset
import torch 
import random
import os 
import numpy as np
import json
import pandas as pd

def load_imdb_dataset(data_path, seed=123):
    """Loads the IMDb movie reviews sentiment analysis dataset.

    # Arguments
        data_path: string, path to the data directory.
        seed: int, seed for randomizer.

    # Returns
        A tuple of training, unlabeled, and test data.

    # References
        Mass et al., http://www.aclweb.org/anthology/P11-1015

        Download and uncompress archive from:
        http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
    """
    imdb_data_path = os.path.join(data_path, 'aclImdb')

    # Load the training data
    train_texts = []
    train_labels = []
    for category in ['pos', 'neg']:
        train_path = os.path.join(imdb_data_path, 'train', category)
        for fname in sorted(os.listdir(train_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(train_path, fname), encoding='utf-8') as f:
                    train_texts.append(f.read())
                train_labels.append(0 if category == 'neg' else 1)
                
    # Load the unsupervised data
    unlabeled_texts = []
    unlabeled_labels = [] # all -1
    unsup_path = os.path.join(imdb_data_path, 'train', 'unsup')
    for fname in sorted(os.listdir(unsup_path)):
        if fname.endswith('.txt'):
            with open(os.path.join(unsup_path, fname)) as f:
                unlabeled_texts.append(f.read())
            unlabeled_labels.append(-1)

    # Load the test data.
    test_texts = []
    test_labels = []
    for category in ['pos', 'neg']:
        test_path = os.path.join(imdb_data_path, 'test', category)
        for fname in sorted(os.listdir(test_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(test_path, fname)) as f:
                    test_texts.append(f.read())
                test_labels.append(0 if category == 'neg' else 1)

    # Shuffle the training data and labels.
    random.seed(seed)
    random.shuffle(train_texts)
    random.seed(seed)
    random.shuffle(train_labels)

    return ((train_texts, np.array(train_labels)),
            (unlabeled_texts, np.array(unlabeled_labels)),
            (test_texts, np.array(test_labels)))

def load_sst2_dataset(data_path, seed=123):
    sst2_data_path = os.path.join(data_path, 'SST-2')

    # Load training data -> unlabeled will come from here
    train_texts = []
    train_labels = []
    train_text_path = os.path.join(sst2_data_path, 'sst2_train_sentences.txt')
    with open(train_text_path, encoding='utf-8') as f:
        texts = f.read()
        result = [text for text in texts.split('\n')]
        result = result[:-1]
        train_texts = [element.strip() for element in result]
    
    train_label_path = os.path.join(sst2_data_path, 'sst2_train_labels.txt')
    with open(train_label_path, encoding='utf-8') as f:
        labels = f.read()
        result = [label for label in labels.split('\n')]
        result = result[:-1]
        train_labels = [int(element) for element in result]

    # Load test data
    test_texts = []
    test_labels = []
    test_text_path = os.path.join(sst2_data_path, 'sst2_test_sentences.txt')
    with open(test_text_path, encoding='utf-8') as f:
        texts = f.read()
        result = [text for text in texts.split('\n')]
        result = result[:-1]
        test_texts = [element.strip() for element in result]
    
    test_label_path = os.path.join(sst2_data_path, 'sst2_test_labels.txt')
    with open(test_label_path, encoding='utf-8') as f:
        labels = f.read()
        result = [label for label in labels.split('\n')]
        result = result[:-1]
        test_labels = [int(element) for element in result]

    # Shuffle training data and labels
    random.seed(seed)
    random.shuffle(train_texts)
    random.seed(seed)
    random.shuffle(train_labels)

    return ((train_texts, np.array(train_labels)),
            (test_texts, np.array(test_labels)))

def load_sst5_dataset(data_path, seed=123):
    sst5_data_path = os.path.join(data_path, 'SST-5')

    # Load training data -> unlabeled will come from here
    train_texts = []
    train_labels = []
    train_path = os.path.join(sst5_data_path, 'sst_train.txt')
    with open(train_path, encoding='utf-8') as f:
        texts = f.read()
        result = [text.split('\t') for text in texts.split('\n')]
        result = result[0:-1]
        train_texts = [element[1].strip() for element in result]
        train_labels = [int(element[0][-1])-1 for element in result]

    # Load validation/dev data -> unlabeled will come from here
    validation_texts = []
    validation_labels = []
    validation_path = os.path.join(sst5_data_path, 'sst_dev.txt')
    with open(validation_path, encoding='utf-8') as f:
        texts = f.read()
        result = [text.split('\t') for text in texts.split('\n')]
        result = result[0:-1]
        validation_texts = [element[1].strip() for element in result]
        validation_labels = [int(element[0][-1])-1 for element in result]

    # Load test data
    test_texts = []
    test_labels = []
    test_path = os.path.join(sst5_data_path, 'sst_test.txt')
    with open(test_path, encoding='utf-8') as f:
        texts = f.read()
        result = [text.split('\t') for text in texts.split('\n')]
        result = result[0:-1]
        test_texts = [element[1].strip() for element in result]
        test_labels = [int(element[0][-1])-1 for element in result]

    # Shuffle training data and labels
    random.seed(seed)
    random.shuffle(train_texts)
    random.seed(seed)
    random.shuffle(train_labels)

    # Combine train and validation
    train_texts.extend(validation_texts)
    train_labels.extend(validation_labels)

    return ((train_texts, np.array(train_labels)),
            (test_texts, np.array(test_labels)))

def load_amazon_elec_dataset(data_path, seed=123):
    amazon_elec_data_path = os.path.join(data_path, 'Electronics_5.json')

    texts = []
    labels = []
    with open(amazon_elec_data_path, encoding='utf-8') as f:
        data = f.read()
        result = data.split('\n')
        for i in range(len(result)-1):
            entry = json.loads(result[i])
            if 'reviewText' in entry:
                texts.append(entry['reviewText'])
                labels.append(int(entry['overall'])-1)
    
    # Shuffle training data and labels
    random.seed(seed)
    random.shuffle(texts)
    random.seed(seed)
    random.shuffle(labels)

    return ((texts, np.array(labels)),)

def load_amazon_elec_binary_dataset(data_path, seed=123):
    amazon_elec_data_path = os.path.join(data_path, 'Electronics_5.json')

    texts = []
    labels = []
    with open(amazon_elec_data_path, encoding='utf-8') as f:
        data = f.read()
        result = data.split('\n')
        for i in range(len(result)-1):
            entry = json.loads(result[i])
            if 'reviewText' in entry:
                # original labels from 1 to 5 -> 1 and 2 become negative label, 4 and 5 become positive label, 3 thrown out
                if int(entry['overall']) in (1, 2):
                    texts.append(entry['reviewText'])
                    labels.append(0)
                elif int(entry['overall']) in (4, 5):
                    texts.append(entry['reviewText'])
                    labels.append(1)    
    
    # Shuffle training data and labels
    random.seed(seed)
    random.shuffle(texts)
    random.seed(seed)
    random.shuffle(labels)

    return ((texts, np.array(labels)),)

def load_modified_amazon_elec_binary_dataset(data_path, seed=123):
    amazon_elec_data_path = os.path.join(data_path, 'elec-unlab')

    # Load training data
    train_texts = [] 
    train_labels = []
    train_texts_file = 'elec-25k-train.txt'
    train_labels_file = 'elec-25k-train.cat' # labels are 1 and 2
    with open(os.path.join(amazon_elec_data_path, train_texts_file), encoding='utf-8') as f:
        train_texts = [element.strip() for element in f.read().split('\n')]
    train_texts = train_texts[:-1]
    with open(os.path.join(amazon_elec_data_path, train_labels_file), encoding='utf-8') as f:
        labels = [element for element in f.read().split('\n')]
        labels = labels[:-1]
        train_labels = [int(element) - 1 for element in labels]

    # Load unlabeled data
    unlabeled_texts = [] 
    unlabeled_labels = []
    unlabeled_texts_files = ['elec-25k-unlab00.txt', 'elec-25k-unlab01.txt']
    for file in unlabeled_texts_files:
        with open(os.path.join(amazon_elec_data_path, file), encoding='utf-8') as f:
            texts = [element.strip() for element in f.read().split('\n')]
        texts = texts[:-1]
        unlabeled_texts.extend(texts)
    unlabeled_labels = [-1 for _ in range(len(unlabeled_texts))]

    # Load test data
    test_texts = [] 
    test_labels = []
    test_texts_file = 'elec-test.txt'
    test_labels_file = 'elec-test.cat' # labels are 1 and 2
    with open(os.path.join(amazon_elec_data_path, test_texts_file), encoding='utf-8') as f:
        test_texts = [element.strip() for element in f.read().split('\n')]
    test_texts = test_texts[:-1]
    with open(os.path.join(amazon_elec_data_path, test_labels_file), encoding='utf-8') as f:
        labels = [element for element in f.read().split('\n')]
        labels = labels[:-1]
        test_labels = [int(element) - 1 for element in labels]

    # Shuffle the training data and labels.
    random.seed(seed)
    random.shuffle(train_texts)
    random.seed(seed)
    random.shuffle(train_labels)

    return ((train_texts, np.array(train_labels)),
            (unlabeled_texts, np.array(unlabeled_labels)),
            (test_texts, np.array(test_labels)))

def load_dbpedia_dataset(data_path, seed=123):
    dbpedia_data_path = os.path.join(data_path, 'dbpedia_csv')

    # Load train data
    train_texts = []
    train_labels = []
    train_path = os.path.join(dbpedia_data_path, 'train.csv')
    train_data = pd.read_csv(train_path, header=None)
    train_data.columns = ['class', 'title', 'content']
    train_texts = train_data['content'].tolist()
    train_labels = train_data['class'].tolist()
    train_labels = [element - 1 for element in train_labels]

    # Load test data
    test_texts = []
    test_labels = []
    test_path = os.path.join(dbpedia_data_path, 'test.csv')
    test_data = pd.read_csv(test_path, header=None)
    test_data.columns = ['class', 'title', 'content']
    test_texts = test_data['content'].tolist()
    test_labels = test_data['class'].tolist()
    test_labels = [element - 1 for element in test_labels]

    # Shuffle training data and labels
    random.seed(seed)
    random.shuffle(train_texts)
    random.seed(seed)
    random.shuffle(train_labels)

    return ((train_texts, np.array(train_labels)),
            (test_texts, np.array(test_labels)))

def load_ag_news_dataset(data_path, seed=123):
    ag_news_data_path = os.path.join(data_path, 'ag_news_csv')

    # Load train data
    train_texts = []
    train_labels = []
    train_path = os.path.join(ag_news_data_path, 'train.csv')
    train_data = pd.read_csv(train_path, header=None)
    train_data.columns = ['class', 'title', 'content']
    train_texts = train_data['content'].tolist()
    train_labels = train_data['class'].tolist()
    train_labels = [element - 1 for element in train_labels]

    # Load test data
    test_texts = []
    test_labels = []
    test_path = os.path.join(ag_news_data_path, 'test.csv')
    test_data = pd.read_csv(test_path, header=None)
    test_data.columns = ['class', 'title', 'content']
    test_texts = test_data['content'].tolist()
    test_labels = test_data['class'].tolist()
    test_labels = [element - 1 for element in test_labels]

    # Shuffle training data and labels
    random.seed(seed)
    random.shuffle(train_texts)
    random.seed(seed)
    random.shuffle(train_labels)

    return ((train_texts, np.array(train_labels)),
            (test_texts, np.array(test_labels)))

def load_yelp_full_dataset(data_path, seed=123):
    yelp_data_path = os.path.join(data_path, 'yelp_review_full_csv')

    # Load train data
    train_texts = []
    train_labels = []
    train_path = os.path.join(yelp_data_path, 'train.csv')
    train_data = pd.read_csv(train_path, header=None)
    train_data.columns = ['class', 'text']
    train_texts = train_data['text'].tolist()
    train_labels = train_data['class'].tolist()
    train_labels = [element - 1 for element in train_labels]

    # Load test data
    test_texts = []
    test_labels = []
    test_path = os.path.join(yelp_data_path, 'test.csv')
    test_data = pd.read_csv(test_path, header=None)
    test_data.columns = ['class', 'text']
    test_texts = test_data['text'].tolist()
    test_labels = test_data['class'].tolist()
    test_labels = [element - 1 for element in test_labels]

    # Shuffle training data and labels
    random.seed(seed)
    random.shuffle(train_texts)
    random.seed(seed)
    random.shuffle(train_labels)

    return ((train_texts, np.array(train_labels)),
            (test_texts, np.array(test_labels)))

def load_yelp_polarity_dataset(data_path, seed=123):
    yelp_data_path = os.path.join(data_path, 'yelp_review_polarity_csv')

    # Load train data
    train_texts = []
    train_labels = []
    train_path = os.path.join(yelp_data_path, 'train.csv')
    train_data = pd.read_csv(train_path, header=None)
    train_data.columns = ['class', 'text']
    train_texts = train_data['text'].tolist()
    train_labels = train_data['class'].tolist()
    train_labels = [element - 1 for element in train_labels]

    # Load test data
    test_texts = []
    test_labels = []
    test_path = os.path.join(yelp_data_path, 'test.csv')
    test_data = pd.read_csv(test_path, header=None)
    test_data.columns = ['class', 'text']
    test_texts = test_data['text'].tolist()
    test_labels = test_data['class'].tolist()
    test_labels = [element - 1 for element in test_labels]

    # Shuffle training data and labels
    random.seed(seed)
    random.shuffle(train_texts)
    random.seed(seed)
    random.shuffle(train_labels)

    return ((train_texts, np.array(train_labels)),
            (test_texts, np.array(test_labels)))

def load_amazon_full_dataset(data_path, seed=123):
    amazon_data_path = os.path.join(data_path, 'amazon_review_full_csv')

    # Load train data
    train_texts = []
    train_labels = []
    train_path = os.path.join(amazon_data_path, 'train.csv')
    train_data = pd.read_csv(train_path, header=None)
    train_data.columns = ['class', 'title', 'text']
    train_texts = train_data['text'].tolist()
    train_labels = train_data['class'].tolist()
    train_labels = [element - 1 for element in train_labels]

    # Load test data
    test_texts = []
    test_labels = []
    test_path = os.path.join(amazon_data_path, 'test.csv')
    test_data = pd.read_csv(test_path, header=None)
    test_data.columns = ['class', 'title', 'text']
    test_texts = test_data['text'].tolist()
    test_labels = test_data['class'].tolist()
    test_labels = [element - 1 for element in test_labels]

    # Shuffle training data and labels
    random.seed(seed)
    random.shuffle(train_texts)
    random.seed(seed)
    random.shuffle(train_labels)

    return ((train_texts, np.array(train_labels)),
            (test_texts, np.array(test_labels)))

def load_amazon_polarity_dataset(data_path, seed=123):
    amazon_data_path = os.path.join(data_path, 'amazon_review_polarity_csv')

    # Load train data
    train_texts = []
    train_labels = []
    train_path = os.path.join(amazon_data_path, 'train.csv')
    train_data = pd.read_csv(train_path, header=None)
    train_data.columns = ['class', 'title', 'text']
    train_texts = train_data['text'].tolist()
    train_labels = train_data['class'].tolist()
    train_labels = [element - 1 for element in train_labels]

    # Load test data
    test_texts = []
    test_labels = []
    test_path = os.path.join(amazon_data_path, 'test.csv')
    test_data = pd.read_csv(test_path, header=None)
    test_data.columns = ['class', 'title', 'text']
    test_texts = test_data['text'].tolist()
    test_labels = test_data['class'].tolist()
    test_labels = [element - 1 for element in test_labels]

    # Shuffle training data and labels
    random.seed(seed)
    random.shuffle(train_texts)
    random.seed(seed)
    random.shuffle(train_labels)

    return ((train_texts, np.array(train_labels)),
            (test_texts, np.array(test_labels)))

def load_yahoo_answers_dataset(data_path, seed=123):
    yahoo_answers_data_path = os.path.join(data_path, 'yahoo_answers_csv')

    # Load train data
    train_texts = []
    train_labels = []
    train_path = os.path.join(yahoo_answers_data_path, 'train.csv')
    train_data = pd.read_csv(train_path, header=None)
    train_data.columns = ['class', 'title', 'content', 'answer']
    train_data = train_data[['class', 'answer']]
    train_data = train_data.dropna()
    train_texts = train_data['answer'].tolist()
    train_labels = train_data['class'].tolist()
    train_labels = [element - 1 for element in train_labels]

    # Load test data
    test_texts = []
    test_labels = []
    test_path = os.path.join(yahoo_answers_data_path, 'test.csv')
    test_data = pd.read_csv(test_path, header=None)
    test_data.columns = ['class', 'title', 'content', 'answer']
    test_data = test_data[['class', 'answer']]
    test_data = test_data.dropna()
    test_texts = test_data['answer'].tolist()
    test_labels = test_data['class'].tolist()
    test_labels = [element - 1 for element in test_labels]

    # Shuffle training data and labels
    random.seed(seed)
    random.shuffle(train_texts)
    random.seed(seed)
    random.shuffle(train_labels)

    return ((train_texts, np.array(train_labels)),
            (test_texts, np.array(test_labels)))

def load_twenty_news_dataset(data_path, seed=123):
    twenty_news_data_path = os.path.join(data_path, '20news-18828')

    texts = []
    labels = []
    label_index = 0

    for folder_name in sorted(os.listdir(twenty_news_data_path)):
        label_index += 1
        path = os.path.join(twenty_news_data_path, folder_name)
        for filename in sorted(os.listdir(path)):
            with open(os.path.join(path, filename), 'rb') as f:
                text = f.read().decode('utf-8', 'ignore')
                texts.append(text)
                labels.append(label_index)

    # Shuffle training data and labels
    random.seed(seed)
    random.shuffle(texts)
    random.seed(seed)
    random.shuffle(labels)

    return ((texts, np.array(labels)),)

def load_airport_tweets_dataset(data_path, seed=123):
    airport_tweets_data_path = os.path.join(data_path, 'Tweets.csv')

    texts = []
    labels = []
    data = pd.read_csv(airport_tweets_data_path)
    texts = data['text'][data['airline_sentiment'] != 'neutral'].tolist()
    labels = data['airline_sentiment'][data['airline_sentiment'] != 'neutral'].tolist()
    labels = [1 if element == 'positive' else 0 for element in labels]

    # Shuffle training data and labels
    random.seed(seed)
    random.shuffle(texts)
    random.seed(seed)
    random.shuffle(labels)

    return ((texts, np.array(labels)),)

class TextDataset(Dataset):
    def __init__(self, txt, labels):
        self.labels = labels
        self.text = txt
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        label = self.labels[idx]
        text = self.text[idx]
        sample = {"Text": text, "Class": label}
        return sample
    def get_labels(self):
        return self.labels
    def get_text(self):
        return self.text

def create_dataset(text, labels, slice_start=None, slice_end=None):
    '''
    Create instance of TextDataset given a dataset
    '''
    if slice_start is None:
        slice_start = 0
    if slice_end is None:
        slice_end = len(text)
    return TextDataset(text[slice_start:slice_end], labels[slice_start:slice_end])

def split_datasets(train, 
                   labeled_proportion, 
                   validation_proportion=None, 
                   test=None, 
                   unlabeled=None, 
                   balance_classes=False, 
                   no_calibration=False):
    '''
    Takes as input three tuples - (train_text, train_labels), (test_text, test_labels), (unlabeled_text, unlabeled_labels)
    Outputs (train_text, train_labels), (validation_text, validation_labels), (test_text, test_labels), (unlabeled_text, unlabeled_labels)
    Test is unchanged
    Validation comes from input train (size = validation_proportion * size of input train)
    Train is a subset of input train (size = labeled_proportion * size of input train)
    Remaining unused samples from input train added to unlabeled set
    labeled_proportion + validation_proportion must be <= 1
    '''
    if no_calibration is False:
        if labeled_proportion + validation_proportion > 1:
            raise Exception('labeled_proportion and validation_proportion cannot sum to more than 1')
    else:
        if labeled_proportion > 1:
            raise Exception('labeled_proportion cannot be greater than 1')

    original_train_length = len(train[1])

    if balance_classes: # create train set with equal number of samples for each class
        unique_classes = np.sort(np.unique(np.array(train[1])))

        train_labels = np.array(train[1])
        train_texts = np.array(train[0])
        idx = np.argsort(train_labels)

        train_labels = list(np.array(train_labels)[idx])
        train_texts = list(np.array(train_texts)[idx])

        new_train_size = int(labeled_proportion * len(train[1]))
        num_per_class = int(1.0 * new_train_size / len(unique_classes)) # rounds down

        idx_for_each_class = []
        for class_label in unique_classes:
            class_idx = train_labels.index(class_label)
            idx_for_each_class.append(class_idx)
        idx_for_each_class.append(len(train_labels))
        
        new_train_texts = []
        new_train_labels = []
        old_train_texts = []
        old_train_labels = []

        for i in range(len(idx_for_each_class)-1):
            index = idx_for_each_class[i]
            next_index = idx_for_each_class[i+1]

            new_train_texts.extend(train_texts[index:index+num_per_class])
            new_train_labels.extend(train_labels[index:index+num_per_class])
            old_train_texts.extend(train_texts[index+num_per_class:next_index])
            old_train_labels.extend(train_labels[index+num_per_class:next_index])


        # shuffle
        random.seed(123)
        zipped_new_train = list(zip(new_train_texts, new_train_labels))
        random.shuffle(zipped_new_train)
        new_train_texts, new_train_labels = zip(*zipped_new_train)

        zipped_old_train = list(zip(old_train_texts, old_train_labels))
        random.shuffle(zipped_old_train)
        old_train_texts, old_train_labels = zip(*zipped_old_train)

        new_train = list(new_train_texts), list(new_train_labels) 
        old_train = list(old_train_texts), list(old_train_labels)
    else:
        new_train_size = int(labeled_proportion * len(train[1]))
        new_train = train[0][:new_train_size], train[1][:new_train_size]
        old_train = train[0][new_train_size:], train[1][new_train_size:]

    if test is None:
        # make test set 20% of original train
        new_test_size = int(0.2 * original_train_length)
        test = old_train[0][:new_test_size], old_train[1][:new_test_size]
        old_train = old_train[0][new_test_size:], old_train[1][new_test_size:]

    if no_calibration is False:
        validation_size = int(validation_proportion * original_train_length)
        validation = old_train[0][:validation_size], old_train[1][:validation_size]
        old_train = old_train[0][validation_size:], old_train[1][validation_size:]
    else:
        validation = None

    if unlabeled is None:
        new_unlabeled_text = np.array(old_train[0])
    else:
        new_unlabeled_text = np.concatenate((unlabeled[0], old_train[0]), axis=0)
    new_unlabeled_labels = np.full((new_unlabeled_text.shape[0], 1), -1)
    new_unlabeled = (new_unlabeled_text, new_unlabeled_labels)
    return new_train, validation, test, new_unlabeled

def dataset_metrics(dataset):
    '''
    Assumes dataset is tuple with (text, labels)
    '''
    text, labels = dataset
    num_samples = len(labels)

    num_classes = len(np.unique(labels))

    # label distribution (# samples / class)
    (unique_label, counts_label) = np.unique(labels, return_counts=True)
    label_distribution = np.asarray((unique_label, counts_label)).T # class number to # samples in that class

    # number of words per sample
    num_words_per_sample = [len(element.split(' ')) for element in text]
    average_words_per_sample = np.average(num_words_per_sample)

    (unique_word, counts_word) = np.unique(num_words_per_sample, return_counts=True)
    word_distribution = np.asarray((unique_word, counts_word)).T # num words to num  samples with that many words

    return num_samples, num_classes, label_distribution, average_words_per_sample, word_distribution

def get_dataset_from_dataloader(dataloader, device):
    texts = []
    labels = []

    with torch.no_grad():
        for (_, batch) in enumerate(dataloader):
            inputs, outputs = batch['Text'].to(device), batch['Class'].to(device)

            texts.append(inputs)
            labels.extend(outputs.cpu().numpy())
        
        texts = torch.cat(texts).cpu().numpy()
    
    return texts, labels
