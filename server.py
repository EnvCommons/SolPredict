from openreward.environments import Server

from solpredict import SolpredictEnvironment

if __name__ == "__main__":
    server = Server([SolpredictEnvironment])
    server.run()
