import flwr as fl
from server.strategy import FederatedStrategy

"""
    (Flow chart)
    
Start Server
        │
Clients connect
        │
Server sends global model
        │
Clients train locally
        │
Clients send weights
        │
Server aggregates (FedAvg)
        │
New global model created
        │
Repeat for 5 rounds

"""


def start_server():

    strategy = FederatedStrategy()

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy
    )


if __name__ == "__main__":
    start_server()
