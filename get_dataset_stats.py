from data_setup import load_ag_news_dataset, load_airport_tweets_dataset, load_amazon_elec_binary_dataset, load_amazon_elec_dataset, load_amazon_full_dataset, load_amazon_polarity_dataset, load_dbpedia_dataset, load_imdb_dataset, load_modified_amazon_elec_binary_dataset, load_sst2_dataset, load_sst5_dataset, load_twenty_news_dataset, load_yahoo_answers_dataset, load_yelp_full_dataset, load_yelp_polarity_dataset, dataset_metrics
import pandas as pd

# script to get statistics on all datasets

binary_classification_datasets = ['imdb', 'sst2', 'amazon_elec_binary', 'modified_amazon_elec_binary', 'yelp_polarity', 'amazon_polarity', 'airport_tweets']
multiclass_classification_datasets = ['sst5', 'amazon_elec', 'dbpedia', 'ag_news', 'yelp_full', 'amazon_full', 'yahoo', 'twenty_news']

dataset_name_to_load_func = {'imdb': load_imdb_dataset, 'sst2': load_sst2_dataset, 'sst5': load_sst5_dataset, 'amazon_elec': load_amazon_elec_dataset, 'amazon_elec_binary': load_amazon_elec_binary_dataset, 'dbpedia': load_dbpedia_dataset, 'ag_news': load_ag_news_dataset, 'yelp_full': load_yelp_full_dataset, 'yelp_polarity': load_yelp_polarity_dataset, 'amazon_full': load_amazon_full_dataset, 'amazon_polarity': load_amazon_polarity_dataset, 'yahoo': load_yahoo_answers_dataset, 'twenty_news': load_twenty_news_dataset, 'airport_tweets': load_airport_tweets_dataset, 'modified_amazon_elec_binary': load_modified_amazon_elec_binary_dataset}

all_datasets = binary_classification_datasets + multiclass_classification_datasets
metrics = pd.DataFrame(columns=['name', 'num_samples', 'num_classes', 'label_distribution', 'average_words_per_sample', 'word_distribution'])

for dataset in all_datasets:

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
    
    train_metrics = dataset_metrics(train)
    if test:
        test_metrics = dataset_metrics(test)
    if unlabeled:
        unlabeled_metrics = dataset_metrics(unlabeled)
    
    metrics.loc[len(metrics.index)] = (dataset + '_train',) + train_metrics 
    if test:
        metrics.loc[len(metrics.index)] = (dataset + '_test',) + test_metrics 
    if unlabeled:
        metrics.loc[len(metrics.index)] = (dataset + '_unlabeled',) + unlabeled_metrics 
metrics.to_csv('dataset_metrics.csv')

