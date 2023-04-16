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