from torch.utils.data import Dataset
import random
import os 
import numpy as np

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
    if slice_start is None:
        slice_start = 0
    if slice_end is None:
        slice_end = len(text)
    return TextDataset(text[slice_start:slice_end], labels[slice_start:slice_end])

def split_datasets(train, test, unlabeled, labeled_proportion, validation_proportion):
    '''
    Takes as input three tuples - (train_text, train_labels), (test_text, test_labels), (unlabeled_text, unlabeled_labels)
    Outputs (train_text, train_labels), (validation_text, validation_labels), (test_text, test_labels), (unlabeled_text, unlabeled_labels)
    Test is unchanged
    Validation comes from input train (size = validation_proportion * size of input train)
    Train is a subset of input train (size = labeled_proportion * size of input train)
    Remaining unused samples from input train added to unlabeled set
    labeled_proportion + validation_proportion must be <= 1
    '''
    if labeled_proportion + validation_proportion > 1:
        raise Exception('labeled_proportion and validation_proportion cannot sum to more than 1')
    new_train_size = int(labeled_proportion * len(train[1]))
    validation_size = int(validation_proportion * len(train[1]))

    new_train = train[0][:new_train_size], train[1][:new_train_size]
    validation = train[0][new_train_size:new_train_size + validation_size], train[1][new_train_size: new_train_size + validation_size]

    new_unlabeled_text = np.concatenate((unlabeled[0], train[0][new_train_size + validation_size:]), axis=0)
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



