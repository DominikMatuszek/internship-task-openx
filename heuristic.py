import numpy as np

def classify_one(data):
    # We will classify cover types based on median elevations of each cover type (it's got about 45.3% accuracy)

    medians = np.array([3146.0, 2935.0, 2404.0, 2231.0, 2796.0, 2428.0, 3363.0]) 
    # ^ Those medians are calculated in the analyze_data.py script 

    elevation = data[0]

    # Find the cover type with the closest median elevation
    answer = np.argmin(np.abs(medians - elevation)) + 1 # +1 because cover types are 1-indexed

    return answer

class HeuristicClassifier:
    def predict(self, data):
        return np.array([classify_one(d) for d in data])


if __name__ == "__main__":
    # Small test to check if the heuristic is working correctly
    
    from load_data import load_data, split_to_features_and_labels
    from sklearn.metrics import accuracy_score, balanced_accuracy_score

    def classify_all(data):
        return np.array([classify_one(d) for d in data])

    data = load_data("covtype.data")
    features, labels = split_to_features_and_labels(data)

    predictions = classify_all(features)

    print(predictions.shape)
    print(labels.shape)

    print("Accuracy: ", accuracy_score(labels, predictions))
    print("Balanced accuracy: ", balanced_accuracy_score(labels, predictions))