import os
import re
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from collections import defaultdict
from mmsdk import mmdatasdk as md
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from constants import SDK_PATH, DATA_PATH, WORD_EMB_PATH, CACHE_PATH
from SingleEncoderModelText import SingleEncoderModelText


def initialize_sdk():
    """Initialize the SDK and check if the path is set correctly."""
    if SDK_PATH is None:
        raise ValueError("SDK path is not specified! Please specify first.")
    sys.path.append(SDK_PATH)
    print(f"SDK path is set to {SDK_PATH}")


def setup_data():
    """Sets up the dataset by downloading and preparing features."""
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    DATASETMD = md.cmu_mosi

    # Download dataset features if not already downloaded
    for feature in [DATASETMD.highlevel, DATASETMD.raw, DATASETMD.labels]:
        try:
            md.mmdataset(feature, DATA_PATH)
        except RuntimeError:
            print(f"{feature} already downloaded.")
    
    return DATASETMD


def load_features(dataset, data_fields):
    """Loads and aligns dataset features from various modalities."""
    print("Loading dataset features...")
    recipe = {field: os.path.join(DATA_PATH, field) + '.csd' for field in data_fields}
    dataset = md.mmdataset(recipe)

    dataset.align(data_fields['text'], collapse_functions=[np.average])  # Average feature alignment
    return dataset


def preprocess_data(dataset, fields, splits, word2id):
    """Preprocess data by normalizing, splitting, and preparing for training."""
    label_field = 'CMU_MOSI_Opinion_Labels'
    label_recipe = {label_field: os.path.join(DATA_PATH, label_field) + '.csd'}
    dataset.add_computational_sequences(label_recipe)

    # Data splits
    train_split = splits['train']
    dev_split = splits['dev']
    test_split = splits['test']

    print(f"Data splits: train {len(train_split)}, dev {len(dev_split)}, test {len(test_split)}")

    # Initialize lists for data and label storage
    train, dev, test = [], [], []
    num_drop = 0

    # Process each data segment
    for segment in dataset[label_field].keys():
        # Extract features and labels
        vid = re.search('(.*)\[.*\]', segment).group(1)
        label = dataset[label_field][segment]['features']
        features = {field: dataset[field][segment]['features'] for field in fields}

        # Check for consistency of feature dimensions
        if not all(feature.shape[0] == features[fields[0]].shape[0] for feature in features.values()):
            num_drop += 1
            continue

        # Normalize features and store
        processed_data = process_features(features)

        # Split into train, dev, test sets
        if vid in train_split:
            train.append((processed_data, label, segment))
        elif vid in dev_split:
            dev.append((processed_data, label, segment))
        elif vid in test_split:
            test.append((processed_data, label, segment))

    print(f"Dropped {num_drop} inconsistent datapoints.")
    print(f"Vocabulary size: {len(word2id)}")
    return train, dev, test


def process_features(features):
    """Normalize features (visual, acoustic, word vectors) to ensure consistency."""
    EPS = 1e-8
    processed = {}
    for key, feature in features.items():
        std_dev = np.std(feature, axis=0, keepdims=True)
        feature = np.nan_to_num((feature - feature.mean(0, keepdims=True)) / (EPS + std_dev))
        feature[:, std_dev.flatten() == 0] = EPS  # Handle zero std deviation
        processed[key] = feature
    return processed


def load_embeddings(w2i, path_to_embedding, embedding_size=300):
    """Load GloVe-like word embeddings and create embedding matrix."""
    emb_mat = np.random.randn(len(w2i), embedding_size)
    with open(path_to_embedding, 'r', encoding='utf-8', errors='replace') as f:
        for line in tqdm(f, total=2196017):
            content = line.strip().split()
            vector = np.asarray(content[-embedding_size:], dtype=float)
            word = ' '.join(content[:-embedding_size])
            if word in w2i:
                emb_mat[w2i[word]] = vector

    emb_mat_tensor = torch.tensor(emb_mat).float()
    torch.save(emb_mat_tensor, 'embedding_matrix.pt')
    print("Embedding matrix saved as 'embedding_matrix.pt'.")
    return emb_mat_tensor


def collate_batch(batch, pad_value):
    """Collate function to handle variable-length sequences in a batch."""
    batch = sorted(batch, key=lambda x: len(x[0][0]), reverse=True)
    sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch], padding_value=pad_value, batch_first=True)
    labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0).float()
    lengths = torch.LongTensor([len(sample[0][0]) for sample in batch])
    return sentences, labels, lengths


def train_model(model, train_loader, dev_loader, max_epoch=1000, patience=8, grad_clip_value=1.0):
    """Train the model with the given data loaders."""
    CUDA = torch.cuda.is_available()
    optimizer = model.create_optimizer(lr=0.001)
    criterion = nn.MSELoss(reduction='sum')
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    lr_scheduler.step()

    if CUDA:
        model.cuda()

    best_valid_loss = float('inf')
    curr_patience = patience
    num_trials = 3

    for e in range(max_epoch):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {e}/{max_epoch}"):
            model.zero_grad()
            t, y, l = batch
            if CUDA:
                t, y, l = t.cuda(), y.cuda(), l.cuda()
            y_tilde = model(t, l)
            loss = criterion(y_tilde, y)
            loss.backward()
            torch.nn.utils.clip_grad_value_([param for param in model.parameters() if param.requires_grad], grad_clip_value)
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {e} - Training loss: {avg_train_loss:.4f}")

        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch in dev_loader:
                t, y, l = batch
                if CUDA:
                    t, y, l = t.cuda(), y.cuda(), l.cuda()
                y_tilde = model(t, l)
                loss = criterion(y_tilde, y)
                valid_loss += loss.item()

        avg_valid_loss = valid_loss / len(dev_loader)
        print(f"Epoch {e} - Validation loss: {avg_valid_loss:.4f}")

        if avg_valid_loss <= best_valid_loss:
            best_valid_loss = avg_valid_loss
            print("New best model found! Saving...")
            torch.save(model.state_dict(), 'best_model.pth')
            torch.save(optimizer.state_dict(), 'best_optimizer.pth')
            curr_patience = patience
        else:
            curr_patience -= 1
            if curr_patience <= 0:
                print("Early stopping due to lack of improvement.")
                break

    return model


def test_model(model, test_loader):
    """Test the trained model on the test dataset."""
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    y_true = []
    y_pred = []
    criterion = nn.MSELoss(reduction='sum')

    with torch.no_grad():
        test_loss = 0.0
        for batch in test_loader:
            t, y, l = batch
            y_tilde = model(t, l)
            loss = criterion(y_tilde, y)
            test_loss += loss.item()

            y_true.append(y.cpu().numpy())
            y_pred.append(y_tilde.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    print(f"Test set performance (Average Loss): {avg_test_loss:.4f}")

    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy


def build():
    """Build, train, and evaluate the model."""
    initialize_sdk()
    datasetMD = setup_data()

    data_fields = {
        'text': 'CMU_MOSI_TimestampedWords',
        'wordvectors': 'CMU_MOSI_TimestampedWordVectors',
        'visual': 'CMU_MOSI_Visual_Facet_41',
        'acoustic': 'CMU_MOSI_COVAREP'
    }
    dataset = load_features(datasetMD, data_fields)
    
    splits = {
        'train': datasetMD.standard_folds.standard_train_fold,
        'dev': datasetMD.standard_folds.standard_valid_fold,
        'test': datasetMD.standard_folds.standard_test_fold
    }

    word2id = defaultdict(lambda: len(word2id))
    train, dev, test = preprocess_data(dataset, data_fields, splits, word2id)

    # Load word embeddings if available
    pretrained_emb = None
    if WORD_EMB_PATH:
        pretrained_emb = load_embeddings(word2id, WORD_EMB_PATH)

    # Initialize model
    model = SingleEncoderModelText(
        dic_size=len(word2id),
        use_glove=True,
        encoder_size=300,
        num_layers=2,
        hidden_dim=128,
        dr=0.2,
        output_size=1
    )

    # Prepare data loaders
    batch_size = 56
    train_loader = DataLoader(train, shuffle=True, batch_size=batch_size, collate_fn=lambda batch: collate_batch(batch, word2id['<pad>']))
    dev_loader = DataLoader(dev, shuffle=False, batch_size=batch_size * 3, collate_fn=lambda batch: collate_batch(batch, word2id['<pad>']))
    test_loader = DataLoader(test, shuffle=False, batch_size=batch_size * 3, collate_fn=lambda batch: collate_batch(batch, word2id['<pad>']))

    # Train the model
    trained_model = train_model(model, train_loader, dev_loader)

    # Test the model
    accuracy = test_model(trained_model, test_loader)
    return trained_model, accuracy


if __name__ == "__main__":
    model, test_accuracy = build()
    print(f"Model training completed with test accuracy: {test_accuracy:.4f}")
