from typing import Optional
import sys
import os
import re
import random
import numpy as np
import torch
import torch.nn as nn
import sys
import mmsdk


from torch import optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm
from mmsdk import mmdatasdk as md
from sklearn.metrics import accuracy_score
from constants import SDK_PATH, DATA_PATH, WORD_EMB_PATH, CACHE_PATH
from SingleEncoderModelAudio import SingleEncoderModelAudio
from mmsdk.mmdatasdk.dataset.standard_datasets.CMU_MOSI.cmu_mosi_std_folds import standard_train_fold, standard_valid_fold, standard_test_fold

def initialize_sdk():
    if SDK_PATH is None:
        print("SDK path is not specified! Please specify first.")
        exit(0)
    sys.path.append(SDK_PATH)
    print(f"SDK path is set to {SDK_PATH}")

def setup_data():
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    DATASETMD = md.cmu_mosi

    try:
        md.mmdataset(DATASETMD.highlevel, DATA_PATH)
    except RuntimeError:
        print("High-level features already downloaded.")

    try:
        md.mmdataset(DATASETMD.raw, DATA_PATH)
    except RuntimeError:
        print("Raw data already downloaded.")

    try:
        md.mmdataset(DATASETMD.labels, DATA_PATH)
    except RuntimeError:
        print("Labels already downloaded.")

    return DATASETMD

def load_features(DATASET):
    visual_field = 'CMU_MOSI_Visual_Facet_41'
    acoustic_field = 'CMU_MOSI_COVAREP'
    text_field = 'CMU_MOSI_TimestampedWords'
    wordvectors_field = 'CMU_MOSI_TimestampedWordVectors'

    recipe = {
        text_field: os.path.join(DATA_PATH, text_field) + '.csd',
        wordvectors_field: os.path.join(DATA_PATH, wordvectors_field) + '.csd',
        visual_field: os.path.join(DATA_PATH, visual_field) + '.csd',
        acoustic_field: os.path.join(DATA_PATH, acoustic_field) + '.csd'
    }

    print("Loading dataset features...")
    dataset = md.mmdataset(recipe)

    def avg(intervals: np.array, features: np.array) -> np.array:
        try:
            return np.average(features, axis=0)
        except:
            return features
        
    dataset.align(text_field, collapse_functions=[avg])
    return dataset, visual_field, acoustic_field, text_field, wordvectors_field

def preprocess_data(DATASETMD, dataset, visual_field, acoustic_field, text_field, wordvectors_field):
    label_field = 'CMU_MOSI_Opinion_Labels'
    label_recipe = {label_field: os.path.join(DATA_PATH, label_field) + '.csd'}
    dataset.add_computational_sequences(label_recipe, destination=None)
    dataset.align(label_field)

    
    train_split = DATASETMD.standard_folds.standard_train_fold
    dev_split = DATASETMD.standard_folds.standard_valid_fold
    test_split = DATASETMD.standard_folds.standard_test_fold
    # train_split = standard_train_fold
    # dev_split = standard_valid_fold
    # test_split = standard_test_fold
    

    print(f"lengths: train {len(train_split)}, dev {len(dev_split)}, test {len(test_split)}\n")
    print(train_split)
    print(dev_split)
    print(test_split)


    word2id = defaultdict(lambda: len(word2id))
    UNK = word2id['<unk>']
    PAD = word2id['<pad>']
    pattern = re.compile('(.*)\[.*\]')

    train, dev, test = [], [], []
    EPS = 1e-8
    num_drop = 0

    for segment in dataset[label_field].keys():
        vid = re.search(pattern, segment).group(1)
        label = dataset[label_field][segment]['features']
        _words = dataset[text_field][segment]['features']
        _visual = dataset[visual_field][segment]['features']
        _acoustic = dataset[acoustic_field][segment]['features']
        _wordvectors = dataset[wordvectors_field][segment]['features']

        if not (_words.shape[0] == _visual.shape[0] == _acoustic.shape[0] == _wordvectors.shape[0]):
            num_drop += 1
            continue

        words, visual, acoustic, wordvectors = [], [], [], []
        for i, word in enumerate(_words):
            if word[0] != b'sp':
                words.append(word2id[word[0].decode('utf-8')])
                visual.append(_visual[i, :])
                acoustic.append(_acoustic[i, :])
                wordvectors.append(_wordvectors[i, :])

        words = np.asarray(words)
        visual = np.asarray(visual)
        acoustic = np.asarray(acoustic)
        wordvectors = np.asarray(wordvectors)


        std_dev_visual = np.std(visual, axis=0, keepdims=True)
        visual = np.nan_to_num((visual - visual.mean(0, keepdims=True)) / (EPS + std_dev_visual))
        visual[:, std_dev_visual.flatten() == 0] = EPS  # Safeguard for zero standard deviation

        # Z-normalization for acoustic modality (across all acoustic features)
        acoustic_mean = np.nanmean(acoustic, axis=0, keepdims=True)
        std_dev_acoustic = np.nanstd(acoustic, axis=0, keepdims=True)
        std_dev_acoustic = np.nan_to_num(std_dev_acoustic)
        std_dev_acoustic[std_dev_acoustic == 0] = EPS  # Safeguard for zero standard deviation

        acoustic = np.nan_to_num((acoustic - acoustic_mean) / (EPS + std_dev_acoustic))

        # Z-normalization for word vectors
        wordvectors_mean = np.nanmean(wordvectors, axis=0, keepdims=True)
        std_dev_wordvectors = np.nanstd(wordvectors, axis=0, keepdims=True)
        std_dev_wordvectors = np.nan_to_num(std_dev_wordvectors)
        std_dev_wordvectors[std_dev_wordvectors == 0] = EPS  # Safeguard for zero standard deviation
        wordvectors = np.nan_to_num((wordvectors - wordvectors_mean) / (EPS + std_dev_wordvectors))

        # plot_hist2(wordvectors, acoustic)

        # Ensure no NaN or Inf values in the data
        if np.any(np.isnan(acoustic)) or np.any(np.isinf(acoustic)):
            print(f"Error in acoustic data for segment {vid}")
        if np.any(np.isnan(visual)) or np.any(np.isinf(visual)):
            print(f"Error in visual data for segment {vid}")
        if np.any(np.isnan(words)) or np.any(np.isinf(words)):
            print(f"Error in wordvectors data for segment {vid}")

        if vid in train_split:
            train.append(((words, visual, acoustic, wordvectors), label, segment))
        elif vid in dev_split:
            dev.append(((words, visual, acoustic, wordvectors), label, segment))
        elif vid in test_split:
            test.append(((words, visual, acoustic, wordvectors), label, segment))

    print(f"Dropped {num_drop} inconsistent datapoints.")
    vocab_size = len(word2id)
    print(f"Vocabulary size: {vocab_size}")
    
    def return_unk():
        return UNK
    word2id.default_factory = return_unk

    return train, dev, test, word2id

def multi_collate_acoustic(batch):
    '''
    Collate function for acoustic data only. Batch will be sorted based on the sequence length of the acoustic features.
    '''
    # Sort batch in descending order based on the length of the acoustic feature sequence
    batch = sorted(batch, key=lambda x: x[0][2].shape[0], reverse=True)
    
    # Extract labels and acoustic features from the batch
    labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0).float()
    acoustic = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch], batch_first=True)
    
    # Sequence lengths (useful for RNNs)
    lengths = torch.LongTensor([sample[0][2].shape[0] for sample in batch])
    return acoustic, labels, lengths


def convert_to_sentiment_category(score):
    if score == 3:
        return 'strongly positive'
    elif score >= 2 and score < 3:
        return 'positive'
    elif score >= 1 and score < 2:
        return 'weakly positive'
    elif score < 1 and score > -1:
        return 'neutral'
    elif score <= -1 and score > -2:
        return 'weakly negative'
    elif score <= -2 and score > -3:
        return 'negative'
    elif score == -3:
        return 'strongly negative'
    else:
        return 'unknown'


def train_model(model, train_loader, dev_loader):
    """
    Trains the model using the given training and development data loaders.
    """
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)

    CUDA = torch.cuda.is_available()
    MAX_EPOCH = 1000

    curr_patience = patience = 8
    num_trials = 3
    grad_clip_value = 1.0

    optimizer = model.create_optimizer(lr=0.001)
    print("Optimizer created")

    if CUDA:
        model.cuda()

    criterion = nn.MSELoss(reduction='sum')
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    lr_scheduler.step()

    best_valid_loss = float('inf')

    train_losses = []
    valid_losses = []
    for e in range(MAX_EPOCH):
        model.train()
        train_iter = tqdm(train_loader)
        train_loss = 0.0

        for batch in train_iter:
            model.zero_grad()
            a, y, l = batch
            batch_size = a.size(0)
            if CUDA:
                a = a.cuda()
                y = y.cuda()
                l = l.cuda()

            y_tilde = model(a, l)
            loss = criterion(y_tilde, y)
            loss.backward()
            torch.nn.utils.clip_grad_value_([param for param in model.parameters() if param.requires_grad], grad_clip_value)
            optimizer.step()
            train_iter.set_description(f"Epoch {e}/{MAX_EPOCH}, current batch loss: {round(loss.item()/batch_size, 4)}")
            train_loss += loss.item()
        train_loss = train_loss / len(train_loader)
        train_losses.append(train_loss)
        print(f"Training loss: {round(train_loss, 4)}")

        model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            for batch in dev_loader:
                model.zero_grad()
                a, y, l = batch
                if CUDA:
                    a = a.cuda()
                    y = y.cuda()
                    l = l.cuda()
                y_tilde = model(a, l)
                loss = criterion(y_tilde, y)
                valid_loss += loss.item()

        valid_loss = valid_loss / len(dev_loader)
        valid_losses.append(valid_loss)
        print(f"Validation loss: {round(valid_loss, 4)}")
        print(f"Current patience: {curr_patience}, current trial: {num_trials}.")
        if valid_loss <= best_valid_loss:
            best_valid_loss = valid_loss
            print("Found new best model on dev set!")
            torch.save(model.state_dict(), 'modelaudio.std')
            torch.save(optimizer.state_dict(), 'optimaudio.std')
            curr_patience = patience
        else:
            curr_patience -= 1
            if curr_patience <= -1:
                print("Running out of patience, loading previous best model.")
                num_trials -= 1
                curr_patience = patience
                model.load_state_dict(torch.load('modelaudio.std'))
                optimizer.load_state_dict(torch.load('optimaudio.std'))
                lr_scheduler.step()
                print(f"Current learning rate: {optimizer.state_dict()['param_groups'][0]['lr']}")

        if num_trials <= 0:
            print("Running out of patience, early stopping.")
            break

    return model


def test_model(model, test_loader):
    """
    Tests the model using the given test data loader.
    """
    CUDA = torch.cuda.is_available()
    model.load_state_dict(torch.load('modelaudio.std'))
    print("Model loaded successfully!")

    y_true = []
    y_pred = []
    model.eval()
    with torch.no_grad():
        test_loss = 0.0
        for batch in test_loader:
            a, y, l = batch

            if CUDA:
                a = a.cuda()
                y = y.cuda()
                l = l.cuda()

            y_tilde = model(a, l)
            loss = nn.MSELoss(reduction='sum')(y_tilde, y)
            test_loss += loss.item()
            y_true.append(y.cpu().numpy())
            y_pred.append(y_tilde.cpu().numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)

    y_true_categories = [convert_to_sentiment_category(score) for score in y_true]
    y_pred_categories = [convert_to_sentiment_category(score) for score in y_pred]

    accuracy = accuracy_score(y_true_categories, y_pred_categories)
    print(f"Accuracy: {accuracy}")

    return accuracy


def build():
    """
    Builds and trains the model, then evaluates it on the test set.
    """
    initialize_sdk()
    datasetMD = setup_data()
    dataset, visual_field, acoustic_field, text_field, wordvectors_field = load_features(datasetMD)
    train, dev, test, word2id = preprocess_data(datasetMD, dataset, visual_field, acoustic_field, text_field, wordvectors_field)
    train_loader = DataLoader(train, shuffle=True, batch_size=56, collate_fn=multi_collate_acoustic)
    dev_loader = DataLoader(dev, shuffle=False, batch_size=168, collate_fn=multi_collate_acoustic)
    test_loader = DataLoader(test, shuffle=False, batch_size=168, collate_fn=multi_collate_acoustic)

    input_size = 74
    hidden_sizes = 128
    num_layers = 2
    output_size = 1
    dropout = 0.5

    print("Initializing model...")
    model = SingleEncoderModelAudio(
        input_size=input_size,
        hidden_dim=hidden_sizes,
        num_layers=num_layers,
        dropout_rate=dropout,
        output_size=output_size
    )

    print("Starting training...")
    trained_model = train_model(model, train_loader, dev_loader)

    print("Starting testing...")
    accuracy = test_model(trained_model, test_loader)

    return trained_model, accuracy


if __name__ == "__main__":
    model, test_accuracy = build()
    print(f"Model training completed with test accuracy: {test_accuracy}")
