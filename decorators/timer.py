# IMPORTS
# import external packages
from time import time


def time_func(func):
    def wrapper(*args, **kwargs):
        print(f"\n{func.__name__} execution began...")
        start = time()
        val = func(*args, **kwargs)
        end = time()
        print(f"{func.__name__} execution finished in {end - start} seconds.\n")
        return val
    return wrapper
