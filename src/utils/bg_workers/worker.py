"""
## Description

Defines the `Worker` class responsible for background task execution.
This module encapsulates the logic for consuming tasks from a shared
queue, dynamically handling both synchronous and asynchronous functions,
and managing retry logic upon task failure.

## Parameters

- None (Module level)

## Returns

`None`

## Raises

- None

## Side Effects

- Initializes a module-level logger (`scheduler.worker`).

## Debug Notes

- The `Worker` runs an infinite loop while `running` is True.
- Synchronous functions are offloaded to `asyncio.get_running_loop().run_in_executor`
  to prevent blocking the main event loop.

## Customization

- Adjust retry logic or backoff strategies inside the `Worker` class.
"""

import asyncio
import inspect
import logging

from .queue import Task, TaskQueue

LOG_SOURCE = "system"


logger = logging.getLogger("scheduler.worker")


class Worker:
    """
    ## Description

    Represents a single background worker responsible for continuously polling
    and executing tasks from a shared `TaskQueue`. Supports both synchronous and
    asynchronous execution models safely without blocking the main event loop.
    """

    def __init__(self, worker_id: int, queue: TaskQueue):
        """
        ## Description

        Initializes a new background worker instance with a unique identifier and
        a reference to the shared task queue.

        ## Parameters

        - `worker_id` (`int`)
          - Description: Unique identifier for the worker.
          - Constraints: Must be >= 0.
          - Example: `1`

        - `queue` (`TaskQueue`)
          - Description: The shared queue from which tasks are consumed.
          - Constraints: Must be a valid `TaskQueue` instance.

        ## Returns

        `None`

        ## Raises

        - None

        ## Side Effects

        - Sets up internal state and marks the worker as running.

        ## Debug Notes

        - Does not immediately start processing; call `start()` to actually begin.

        ## Customization

        - Add worker-specific initialization logic like thread-local storage here.
        """
        self.worker_id = worker_id
        self.queue = queue
        self.running = True

    async def start(self):
        """
        ## Description

        Continuously polls the task queue for new tasks and executes them. Automatically
        handles sync vs async functions and manages basic retry logic on failure.

        ## Parameters

        - None

        ## Returns

        `None`

        ## Raises

        - None (Internal exceptions during task execution are caught and logged).

        ## Side Effects

        - Executes arbitrary functions defined in retrieved tasks.
        - May re-enqueue failed tasks if max retries are not exceeded.
        - Emits standard log events for task execution lifecycle and failures.
        - Calls `task_done` on the queue after processing.

        ## Debug Notes

        - Runs an infinite loop while `self.running` is True.
        - Synchronous tasks are offloaded to an executor to avoid blocking the asyncio event loop.
        - A blocked sync task might starve the worker thread pool.

        ## Customization

        - Adjust retry logic, implement exponential backoff strategies, or add dead-letter queues here.
        """
        logger.info(f"Worker {self.worker_id} started")

        while self.running:
            task: Task = await self.queue.get()

            try:
                func = task.func

                logger.info(
                    f"Worker {self.worker_id} running {func.__name__} with params={getattr(task, 'params', {})}"
                )

                # New behavior: Task.parameters are provided as a single dict (`Task.params`)
                # and are always passed to the target function as keyword arguments.
                # This provides a consistent way to supply parameters to both sync and async callables.
                params = getattr(task, "params", {}) or {}

                if inspect.iscoroutinefunction(func):
                    # Pass all parameters as keyword args from Task.params
                    await func(**params)
                else:
                    loop = asyncio.get_running_loop()
                    # Run synchronous functions in the default executor and forward params as kwargs
                    await loop.run_in_executor(None, lambda: func(**params))

            except Exception as e:
                logger.error(f"Task failed: {e}")

                if task.retries < task.max_retries:
                    task.retries += 1
                    await self.queue.put(task)

            finally:
                self.queue.task_done()
