import time
import functools


def timer_decorator(func):
    count = [0]

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        if duration > 1:
            print(f"{func.__name__}#{count[0]}: {(end_time - start_time):.3f} s")
        elif duration > 1e-3:
            print(f"{func.__name__}#{count[0]}: {(end_time - start_time)*1e3:.3f} ms")
        else:
            print(f"{func.__name__}#{count[0]}: {(end_time - start_time)*1e6:.3f} us")
        count[0] += 1
        return result

    return wrapper
