from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore


class BoundedExecutor:
    """ThreadPoolExecutor that blocks on ``submit()`` when the queue is full.

    Keeps memory bounded under burst load without silently dropping work.
    """

    def __init__(self, bound: int, max_workers: int) -> None:
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.semaphore = BoundedSemaphore(bound + max_workers)

    def submit(self, fn, *args, **kwargs):
        self.semaphore.acquire()
        try:
            future = self.executor.submit(fn, *args, **kwargs)
        except Exception:
            self.semaphore.release()
            raise
        future.add_done_callback(lambda _: self.semaphore.release())
        return future

    def shutdown(self, wait: bool = True) -> None:
        self.executor.shutdown(wait)
