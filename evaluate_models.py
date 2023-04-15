from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf 

from heuristic import HeuristicClassifier
from scikit_models import train_knn_model, train_svc
from train_neural_net import get_fully_trained_model
from load_data import load_data, split_to_train_test_val, split_to_features_and_labels 

from matplotlib import pyplot as plt

def evaluate_model(model, test_features, test_labels):
    predictions = model.predict(test_features)

    accuracy = accuracy_score(test_labels, predictions)
    balanced_accuracy = balanced_accuracy_score(test_labels, predictions)
    confusion = confusion_matrix(test_labels, predictions, normalize="true")

    return accuracy, balanced_accuracy, confusion

def main():
    filename = "covtype.data"

    data = load_data(filename)
    
    train_data, val_data, test_data = split_to_train_test_val(data, 0.7, 0.5)

    train_features, train_labels = split_to_features_and_labels(train_data)
    validation_features, validation_labels = split_to_features_and_labels(val_data)
    test_features, test_labels = split_to_features_and_labels(test_data)

    heuristic_model = HeuristicClassifier()
    knn_model = train_knn_model(train_features, train_labels)
    svc_model = train_svc(train_features, train_labels)
    neural_net_model = get_fully_trained_model(train_features, train_labels, validation_features, validation_labels, epochs=10)

    models = [heuristic_model, knn_model, svc_model, neural_net_model]
    model_names = ["Heuristic", "KNN", "SVC", "Neural Net"]

    results = []

    for model in models:
        accuracy, balanced_accuracy, confusion = evaluate_model(model, test_features, test_labels)

        results.append((accuracy, balanced_accuracy, confusion))

    # Create a pyplot figure with 2 subplots
    fig, axs = plt.subplots(1, 2)

    # Plot the accuracy and balanced accuracy for each model
    for i in range(len(models)):
        axs[0].bar(model_names[i], results[i][0])
        axs[1].bar(model_names[i], results[i][1])
    
    # Set the labels for the subplots
    axs[0].set_title("Accuracy")
    axs[1].set_title("Balanced Accuracy")

    # Show the plot
    plt.show()

    plt.clf()

    # Plot the confusion matrices on the same figure (one for each model)

    fig, ax = plt.subplots(2, 2)

    for i, result in enumerate(results):
        confusion = result[2]

        disp = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=[1, 2, 3, 4, 5, 6, 7])
       
        disp.plot(ax=ax[i // 2][i % 2])

        ax[i // 2][i % 2].set_title(model_names[i])

    plt.show()


if __name__ == "__main__":
    main()