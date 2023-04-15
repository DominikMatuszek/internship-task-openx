# A file with bunch of functions to load and preprocess the data
# Please note that the dataset is not being feature engineered at all, as requested 

import pandas as pd
import numpy as np

def load_data(dir_path):
    df = pd.read_csv(dir_path, header=None)
    
    # Convert to a numpy array
    df = df.values

    # Shuffle the data
    np.random.shuffle(df)

    return df

def split_to_features_and_labels(data):
    features = data[:, :-1]
    labels = data[:, -1]
    return features, labels

def split_data(data, split_position=0.8):
    split_position = int(len(data) * split_position)
    train_data = data[:split_position]
    test_data = data[split_position:]

    return train_data, test_data

# Note that test_frac means how much of the rest of the data is used for testing and how much for validation
def split_to_train_test_val(data, train_size=0.7, test_frac=0.5):
    train_data, rest = split_data(data, train_size)
    test_data, val_data = split_data(rest, 1-test_frac)

    return train_data, test_data, val_data

def get_splits(data, test_size=0.2):
    train_data, test_data = split_data(data, 1-test_size)

    train_features, train_labels = split_to_features_and_labels(train_data)
    test_features, test_labels = split_to_features_and_labels(test_data)
    
    return train_features, train_labels, test_features, test_labels

def get_preprocessed_data(dir_path, test_size=0.2):
    data = load_data(dir_path)
    return get_splits(data, test_size)

def main():
    train_features, train_labels, test_features, test_labels = get_preprocessed_data("covtype.data")

    print("Train features shape: ", train_features.shape)
    print("Train labels shape: ", train_labels.shape)
    print("Test features shape: ", test_features.shape)
    print("Test labels shape: ", test_labels.shape)

    print(train_features[0])

if __name__ == '__main__':
    main()