import valkey
import numpy as np
from termcolor import colored

def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class Redis:
    def __init__(self, host='localhost', port=6379, decode_responses=True):
        self.client = valkey.Valkey(host=host, port=port, decode_responses=decode_responses)

        try:
            if self.client.ping():
                print(colored("Connected to Redis", "green"))
        except Exception as e:
            print(colored(f"Failed to connect to Redis: {e}", "red"))

    def get(self, key):
        return self.client.get(key)

    def set(self, key, value):
        return self.client.set(key, value)

    def delete(self, key):
        return self.client.delete(key)

    def exists(self, key):
        return self.client.exists(key)

    def flushdb(self):
        return self.client.flushdb()

    def store_vector(self, emb_np, labels_np, dataset, ttl=600):
        pipe = self.client.pipeline()

        for vec, label in zip(emb_np, labels_np):
            team_number = dataset.classes[int(label)]

            key = f"reid:robot:{team_number}:vector"
            pipe.set(key, vec.tobytes(), ex=ttl)

        return pipe.execute()

    def get_vector(self, team):
        raw = self.client.get(f"vec:robot:{team}")
        return np.frombuffer(raw, dtype=np.float32)