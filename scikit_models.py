# We will perform k-nearest neighbours classification on our dataset
from load_data import get_preprocessed_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score

# This model achieves 0.97 accuracy and 0.93 balanced accuracy, which seems like a good baseline
def train_knn_model(train_features, train_labels):

    model = KNeighborsClassifier(n_neighbors=5, algorithm="ball_tree")
    model.fit(train_features, train_labels)

    return model

def train_svc(train_features, train_labels, num_samples=5000):

    train_features = train_features[:num_samples]
    train_labels = train_labels[:num_samples]

    model = SVC(decision_function_shape="ovr", kernel='rbf', class_weight="balanced", gamma="scale", C=1000) 
    model.fit(train_features, train_labels)

    return model

def main():
    path = "covtype.data"

    train_features, train_labels, test_features, test_labels = get_preprocessed_data(path)

    print("Training knn model")
    train_knn_model(train_features, train_labels)

    print("Training svc model")
    train_svc(train_features, train_labels)

if __name__ == "__main__":
    main()
