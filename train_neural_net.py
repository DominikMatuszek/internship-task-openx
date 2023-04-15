import keras_tuner as kt
from tensorflow import keras
from load_data import split_data, load_data, split_to_features_and_labels, split_to_train_test_val
import numpy as np

from matplotlib import pyplot as plt

# Behaves like a scikit-learn model in terms of predictions, so that it doesn't return probabilities but actual labels 
# Thus, a layer of indirection is needed
class NeuralNetModel:
    def __init__(self, model):
        self.model = model
    
    def predict(self, data):
        return np.argmax(self.model.predict(data), axis=1) + 1 # +1 because cover types are 1-indexed

def build_model(hp):
    num_hidden_layers = hp.Int("num_hidden_layers", 1, 3)

    model = keras.Sequential()

    for i in range(num_hidden_layers):
        model.add(keras.layers.Dense(units=hp.Int(f"units_{i}", 64, 512, 32), activation="relu"))
    
    if hp.Choice("dropout", values=[True, False]):
        dropout_rate = hp.Float("dropout_rate", 0.1, 0.5, step=0.1)
        model.add(keras.layers.Dropout(dropout_rate))
    
    model.add(keras.layers.Dense(7, activation="softmax"))

    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Float("learning_rate", 1e-4, 1e-2, sampling="log")
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

def find_best_model(train_data, train_labels, validation_data, validation_labels):
    tuner = kt.RandomSearch(
        build_model,
        objective="val_loss",
        max_trials=10,
        executions_per_trial=3,
        directory="hyperparameter_tuning",
        overwrite=False, # Set to True if you haven't run this before
        project_name="covtype"
    )

    tuner.search(
        train_data,
        train_labels,
        epochs=5,
        validation_data=(validation_data, validation_labels)
    )

    tuner.results_summary()

    model = tuner.get_best_models(num_models=2)[0]

    return model

def train_model(model, train_data, train_labels, validation_data, validation_labels, test_data, test_labels, epochs=10):
    return model.fit(
        train_data,
        train_labels,
        epochs=epochs,
        validation_data=(validation_data, validation_labels), 
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=2,
                restore_best_weights=True
            ),
        ]
    )


def get_fully_trained_model(train_features, train_labels, validation_features, validation_labels, test_features, test_labels, epochs=10, get_history=False):
    train_labels = np.subtract(train_labels, 1)
    validation_labels = np.subtract(validation_labels, 1)

    model = find_best_model(train_features, train_labels, validation_features, validation_labels)
    train_history = train_model(model, train_features, train_labels, validation_features, validation_labels, test_features, test_labels, epochs=epochs)

    model = NeuralNetModel(model)

    if get_history:
        return model, train_history
    else:
        return model

# Implementation of the task 3
def main():
    data = load_data("covtype.data")

    train_data, test_data, val_data = split_to_train_test_val(data, 0.7, 0.5)

    train_features, train_labels = split_to_features_and_labels(train_data)
    validation_features, validation_labels = split_to_features_and_labels(val_data)
    test_features, test_labels = split_to_features_and_labels(test_data)
    
    model, train_history = get_fully_trained_model(train_features, train_labels, validation_features, validation_labels, test_features, test_labels, get_history=True)

    # We want 2 plots, one for the loss and one for the accuracy
    fig, axs = plt.subplots(1, 2)

    # Plot the loss
    axs[0].plot(train_history.history["loss"], label="Training loss")
    axs[0].plot(train_history.history["val_loss"], label="Validation loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    # Plot the accuracy
    axs[1].plot(train_history.history["accuracy"], label="Training accuracy")
    axs[1].plot(train_history.history["val_accuracy"], label="Validation accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()

    plt.show()

if __name__ == "__main__":
    main()