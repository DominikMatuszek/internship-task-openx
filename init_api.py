# This code should only be ran once, when the API is first initialized.
# It trains all the neural net model and then saves it to the models folder.
# Note that we don't save the KNN and SVC models, as they are rather fast to train.

from load_data import get_preprocessed_data
from train_neural_net import get_fully_trained_model

def main():
    path = "covtype.data"

    # We don't need the test split now, we won't be evaluating the model here 
    train_features, train_labels, validation_features, validation_labels = get_preprocessed_data(path)

    model = get_fully_trained_model(train_features, train_labels, validation_features, validation_labels, epochs=1)

    model.save("models/neural_net_model") 
    
if __name__ == "__main__":
    main()