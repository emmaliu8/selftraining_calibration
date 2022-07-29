import numpy as np
import torch

from data_setup import create_dataset, load_imdb_dataset, load_sst2_dataset, load_sst5_dataset, load_amazon_elec_dataset, load_amazon_elec_binary_dataset, load_dbpedia_dataset, load_ag_news_dataset, load_yelp_full_dataset, load_yelp_polarity_dataset, load_amazon_full_dataset, load_amazon_polarity_dataset, load_yahoo_answers_dataset, load_twenty_news_dataset, load_airport_tweets_dataset
from extract_features import featurize_dataset

batch_size = 256

# script to convert text in datasets to features and save to files

def main(dataset):
    '''
    dataset can be any key from dataset_name_to_load_func
    '''

    # reproducible
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_name_to_load_func = {'imdb': load_imdb_dataset, 'sst2': load_sst2_dataset, 'sst5': load_sst5_dataset, 'amazon_elec': load_amazon_elec_dataset, 'amazon_elec_binary': load_amazon_elec_binary_dataset, 'dbpedia': load_dbpedia_dataset, 'ag_news': load_ag_news_dataset, 'yelp_full': load_yelp_full_dataset, 'yelp_polarity': load_yelp_polarity_dataset, 'amazon_full': load_amazon_full_dataset, 'amazon_polarity': load_amazon_polarity_dataset, 'yahoo': load_yahoo_answers_dataset, 'twenty_news': load_twenty_news_dataset, 'airport_tweets': load_airport_tweets_dataset}

    data = dataset_name_to_load_func[dataset]('../') 

    # split data into train, unlabeled, test
    if len(data) == 1:
        train = data[0]
        test = None 
        unlabeled = None
    elif len(data) == 2:
        train, test = data
        unlabeled = None
    else:
        train, unlabeled, test = data

    # for testing purposes
    # train = train[0][:100], train[1][:100]
    # unlabeled = unlabeled[0][:100], unlabeled[1][:100]
    # test = test[0][:100], test[1][:100]

    # create dataset objects for each split
    train_dataset = create_dataset(train[0], train[1])
    if test is not None:
        test_dataset = create_dataset(test[0], test[1])
    if unlabeled is not None:
        unlabeled_dataset = create_dataset(unlabeled[0], unlabeled[1])

    train = featurize_dataset(train_dataset, device, batch_size, dataset, 'train_data.pt')
    if test is not None:
        test = featurize_dataset(test_dataset, device, batch_size, dataset, 'test_data.pt')
    if unlabeled is not None:
        unlabeled = featurize_dataset(unlabeled_dataset, device, batch_size, dataset, 'unlabeled_data.pt')

# print('amazon_full')
# main('amazon_full')