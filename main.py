from client import Client
from distribution import FixNonIID
from server import Server

from plato.samplers.registry import registered_samplers

registered_samplers["fixed_noniid"] = FixNonIID
def main():
    server = Server()

    client = Client()
    server.run(client=client)


if __name__ == "__main__":
    main()