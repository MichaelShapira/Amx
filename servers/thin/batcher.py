"""Micro-batcher: one inference thread, static batches (design 7.5).

The HTTP event loop submits jobs; the inference thread takes the oldest job, waits until
max_wait_ms after that job arrived or until the batch holds max_batch units (one unit per
generation request, one per input for classify and detect), then runs the batch to
completion. Jobs whose batching key differs from the first job's (different sampling
settings) wait for a later batch, in arrival order.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Job:
    payload: Any
    units: int = 1
    key: tuple = ()
    arrived: float = field(default_factory=time.monotonic)


class MicroBatcher:
    def __init__(self, run: Callable[[list[Job]], None], *, max_batch: int, max_wait_ms: float,
                 on_error: Callable[[list[Job], BaseException], None]):
        self.run, self.on_error = run, on_error
        self.max_batch, self.max_wait = max_batch, max_wait_ms / 1000.0
        self.inbox: queue.Queue[Job | None] = queue.Queue()
        self.pending: deque[Job] = deque()
        self.batches = 0
        self.thread = threading.Thread(target=self._loop, name="inference", daemon=True)
        self.thread.start()

    def submit(self, job: Job) -> None:
        self.inbox.put(job)

    def close(self) -> None:
        self.inbox.put(None)
        self.thread.join(timeout=5)

    def _drain(self, block: bool, timeout: float | None) -> bool:
        """Move jobs from the inbox to pending. Returns False when closed."""
        try:
            item = self.inbox.get(block=block, timeout=timeout)
        except queue.Empty:
            return True
        if item is None:
            return False
        self.pending.append(item)
        while True:
            try:
                item = self.inbox.get_nowait()
            except queue.Empty:
                return True
            if item is None:
                return False
            self.pending.append(item)

    def _take(self) -> list[Job]:
        head = self.pending.popleft()
        batch, units, keep = [head], head.units, deque()
        while self.pending and units < self.max_batch:
            j = self.pending.popleft()
            if j.key == head.key and units + j.units <= self.max_batch:
                batch.append(j)
                units += j.units
            else:
                keep.append(j)
        keep.extend(self.pending)
        self.pending = keep
        return batch

    def _units(self, key: tuple) -> int:
        return sum(j.units for j in self.pending if j.key == key)

    def _loop(self) -> None:
        open_ = True
        while open_ or self.pending:
            if not self.pending:
                if not self._drain(block=True, timeout=None):
                    return
                continue
            head = self.pending[0]
            deadline = head.arrived + self.max_wait
            while open_ and self._units(head.key) < self.max_batch:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                open_ = self._drain(block=True, timeout=left)
            if open_:
                open_ = self._drain(block=False, timeout=None)
            batch = self._take()
            self.batches += 1
            try:
                self.run(batch)
            except BaseException as e:  # noqa: BLE001 - every waiting request must get an answer
                self.on_error(batch, e)
