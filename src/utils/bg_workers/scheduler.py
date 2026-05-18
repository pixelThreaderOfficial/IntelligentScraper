"""
## Description

Provides the `Scheduler` class for managing a pool of background workers.
This module handles the initialization of workers, task enqueuing, and
graceful shutdown procedures for the asynchronous background task system.

## Parameters

- None (Module level)

## Returns

`None`

## Raises

- None

## Side Effects

- Initializes a module-level logger (`scheduler`).

## Debug Notes

- The scheduler must be started explicitly via the `start()` method before
  tasks are processed.
- Uses `asyncio` for concurrency management.

## Customization

- N/A
"""

import asyncio
import logging

from .queue import Task, TaskQueue
from .worker import Worker

LOG_SOURCE = "system"


logger = logging.getLogger("scheduler")


class Scheduler:
    """
    ## Description

    Manages a pool of background workers to process asynchronous and synchronous
    tasks concurrently. Acts as the main entry point for queuing tasks and
    coordinating worker lifecycle.
    """

    def __init__(self, workers: int = 3):
        """
        ## Description

        Initializes the Task Scheduler with a specified number of concurrent workers.
        Sets up the underlying task queue but does not start the workers yet.

        ## Parameters

        - `workers` (`int`, optional)
          - Description: The number of background workers to spawn.
          - Constraints: Must be > 0. Default is 3.
          - Example: `5`

        ## Returns

        `None`

        ## Raises

        - None

        ## Side Effects

        - Initializes an empty `TaskQueue`.
        - Prepares internal state tracking lists.

        ## Debug Notes

        - Does not immediately start processing; `start()` must be called explicitly.

        ## Customization

        - Increase `workers` to handle higher concurrent I/O-bound task loads.
        """
        self.queue = TaskQueue()
        self.worker_count = workers
        self.worker_tasks = []
        self.started = False

    async def start(self):
        """
        ## Description

        Bootstraps the worker pool and begins listening for incoming tasks on the queue.
        Safely ignores consecutive calls if already started.

        ## Parameters

        - None

        ## Returns

        `None`

        ## Raises

        - None

        ## Side Effects

        - Spawns multiple asynchronous worker tasks via `asyncio.create_task`.
        - Sets the `started` flag to True.
        - Emits standard log events.

        ## Debug Notes

        - Workers run indefinitely until `shutdown` or application termination.
        - Ensure an active asyncio event loop is running before calling this.

        ## Customization

        - Add an event hook for successful startup broadcasting.
        """
        if self.started:
            return

        logger.info("Starting scheduler")

        for i in range(self.worker_count):
            worker = Worker(i, self.queue)
            task = asyncio.create_task(worker.start())

            self.worker_tasks.append(task)

        self.started = True

    async def schedule(self, func, params: dict | None = None):
        """
        ## Description

        Enqueues a specific function or coroutine to be executed by an available worker.
        Parameters are provided as a single `params` dictionary (object) for consistency
        across scheduling calls and worker execution.

        ## Parameters

        - `func` (`Callable`)
          - Description: The target function or coroutine to execute.
          - Constraints: Can be synchronous or asynchronous.
          - Example: `fetch_data`

        - `params` (`dict`, optional)
          - Description: Single dictionary containing all parameters to pass to the function
            as keyword arguments. This replaces separate `*args`/`**kwargs` usage and
            keeps task payloads consistent.
          - Constraints: Keys must match the target function's parameter names.
          - Example: `{"name": "job1", "timeout": 10}`

        ## Returns

        `None`

        ## Raises

        - None

        ## Side Effects

        - Appends a new `Task` payload (with `params`) to the internal `TaskQueue`.
        - Logs scheduling initiation.

        ## Debug Notes

        - Synchronous functions are executed in a separate thread pool by the worker.
        - The `Worker` will call the target function with `**params`. If you need to
          pass positional-only arguments, wrap them in the params dict under agreed keys.

        ## Customization

        - Add priority/delay fields inside the `Task.params` dict if extending the Task model.
        """
        task = Task(func=func, params=params or {})

        await self.queue.put(task)

    async def shutdown(self):
        """
        ## Description

        Gracefully waits for the task queue to become completely empty,
        blocking until all currently queued tasks finish execution.

        ## Parameters

        - None

        ## Returns

        `None`

        ## Raises

        - None

        ## Side Effects

        - Blocks execution until internal queue signals `task_done` for all items.
        - Emits a shutdown success log.

        ## Debug Notes

        - This does NOT kill running workers directly; it just waits for the queue to drain.
        - If tasks hang indefinitely, this function will block indefinitely.

        ## Customization

        - Combine with a cancellation signal to forcibly stop workers after joining the queue.
        """
        logger.info("Scheduler shutdown")
        await self.queue.join()
