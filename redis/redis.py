from typing import Any

import valkey
import numpy as np
from numpy import dtype, float64, ndarray
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
    def __init__(self, host='localhost', port=6379, decode_responses=True) -> None:
        self.client = valkey.Valkey(host=host, port=port, decode_responses=decode_responses)

        try:
            if self.client.ping():
                print(colored("Connected to Redis", "green"))
        except Exception as e:
            print(colored(f"Failed to connect to Redis: {e}", "red"))

    def get(self, key: str) -> bytes | None:
        return self.client.get(key)

    def set(self, key: str, value: bytes) -> bool:
        return self.client.set(key, value)

    def delete(self, key: str) -> int:
        return self.client.delete(key)

    def exists(self, key: str) -> bool:
        return self.client.exists(key)

    def flushdb(self) -> bool:
        return self.client.flushdb()

    def update_prototype(self, team_number: int, new_vec_np: ndarray[tuple[Any, ...], dtype[float64]], alpha: float = None) -> bool | None:
        mean_key = f"reid:robot:{team_number}:mean"
        count_key = f"reid:robot:{team_number}:count"

        raw_mean = self.client.get(mean_key)

        if raw_mean is None:
            self.client.set(mean_key, new_vec_np.tobytes())
            self.client.set(count_key, 1)
            return

        old_mean = np.frombuffer(raw_mean, dtype=np.float32)

        if alpha is None:
            # Running mean
            count = int(self.client.get(count_key))
            new_mean = old_mean + (new_vec_np - old_mean) / (count + 1)
            self.client.incr(count_key)
        else:
            # Exponential moving average
            new_mean = alpha * new_vec_np + (1 - alpha) * old_mean

        # Normalize
        new_mean = new_mean / np.linalg.norm(new_mean)

        return self.client.set(mean_key, new_mean.astype(np.float32).tobytes())

    def get_prototype(self, team_number: int) -> None | ndarray[tuple[Any, ...], dtype[float64]]:
        raw = self.client.get(f"reid:robot:{team_number}:mean")
        if raw is None:
            return None
        return np.frombuffer(raw, dtype=np.float32)
