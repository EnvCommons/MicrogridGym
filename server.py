from openreward.environments import Server

from microgrid import MicrogridGym

if __name__ == "__main__":
    server = Server([MicrogridGym])
    server.run()
