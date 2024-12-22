
import threading
import multiprocessing
from functools import wraps

def run_in_thread(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        thread = threading.Thread(target=func, args=args, kwargs=kwargs)
        thread.start()
    return wrapper

def run_in_process(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        process = multiprocessing.Process(target=func, args=args, kwargs=kwargs)
        process.start()
    return wrapper


class SingletonMeta(type):
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

