import flwr as fl
from flwr.common import parameters_to_ndarrays

from models.cnn_model import create_model


# Create global model
server_model = create_model()


class FederatedStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):

        # Perform standard FedAvg aggregation
        aggregated = super().aggregate_fit(rnd, results, failures)

        if aggregated:

            parameters, _ = aggregated

            # Convert Flower parameters to numpy weights
            weights = parameters_to_ndarrays(parameters)

            # Update global model
            server_model.set_weights(weights)

            print(f"\nGlobal model updated at round {rnd}")

            # Save model
            server_model.save("global_model.keras")

        return aggregated

