from client import Client
from server import Server


def main():
    server = Server()

    client = Client()
    server.run(client=client)


if __name__ == "__main__":
    main()