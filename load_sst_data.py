from datasets import load_dataset

# script to load sst dataset (from huggingface) and save to txt files

sst = load_dataset('sst')
train = sst['train']
validation = sst['validation']
test = sst['test']

train_sentences = train[:]['sentence'] + validation[:]['sentence']
test_sentences = test[:]['sentence']
train_labels = train[:]['label'] + validation[:]['label']
test_labels = test[:]['label']
train_labels = [1 if element >= 0.5 else 0 for element in train_labels]
test_labels = [1 if element >= 0.5 else 0 for element in test_labels]

with open('sst2_train_sentences.txt', 'w') as f:
    for item in train_sentences:
        f.write("%s\n" % item)

with open('sst2_train_labels.txt', 'w') as f:
    for item in train_labels:
        f.write("%s\n" % item)

with open('sst2_test_sentences.txt', 'w') as f:
    for item in test_sentences:
        f.write("%s\n" % item)

with open('sst2_test_labels.txt', 'w') as f:
    for item in test_labels:
        f.write("%s\n" % item)

