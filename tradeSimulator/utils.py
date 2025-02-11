import time

class RateLimiter:
    def __init__(self, rate_per_second: int):
        self.rate_per_second = rate_per_second
        self.interval = 1.0 / rate_per_second
        self.next_execution_time = time.monotonic()

    def __enter__(self):
        current_time = time.monotonic()
        if current_time < self.next_execution_time:
            time.sleep(self.next_execution_time - current_time)
        self.next_execution_time += self.interval

    def __exit__(self, exc_type, exc_value, traceback):
        pass
