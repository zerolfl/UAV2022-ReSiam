import time
from functools import wraps

def timethis(num_times=1):
    def repeat(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            for i in range(num_times):
                result = func(*args, **kwargs)
            end = time.time()
            func_name = func.__name__
            consume = (end - start) / num_times
            print(f'{func_name} (x{num_times}) consume avg. time ---> {consume:.6f}')
            return result
        return wrapper
    return repeat