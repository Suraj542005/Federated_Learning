import flwr as fl

from models.cnn_model import create_model
from data.mnist_loader import load_mnist

# Load dataset
x_train, y_train, x_test, y_test = load_mnist()

# Client-1 uses first half of training data
x_train = x_train[:30000]
y_train = y_train[:30000]


class Client1(fl.client.NumPyClient):

    def __init__(self):
        self.model = create_model()

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):

        # Update models with global weights
        self.model.set_weights(parameters)

        # Train locally
        self.model.fit(
            x_train,
            y_train,
            epochs=1,
            batch_size=32,
            verbose=1
        )

        return self.model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):

        self.model.set_weights(parameters)

        loss, accuracy = self.model.evaluate(x_test, y_test)

        return loss, len(x_test), {"accuracy": accuracy}


# Start client
fl.client.start_numpy_client(
    server_address="localhost:8080",
    client=Client1()
)
