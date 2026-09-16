"""Continuous rollout production: one task per prompt group, siblings gathered inside it."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, Set

from unirl.rollout.manager.buffer import BufferEntry, GroupBuffer
from unirl.rollout.manager.filters import Group

if TYPE_CHECKING:
    from unirl.types.sample import Sample

logger = logging.getLogger(__name__)

_IDLE_INTERVAL_S = 0.05
_NO_PROGRESS_WARN_S = 30.0


class Producer:
    """Keeps ``budget`` groups outstanding across flight and buffer; see the rollout README."""

    def __init__(
        self,
        buffer: GroupBuffer,
        *,
        launch: Callable[[int, "Sample"], Any],
        pull: Callable[[], Optional["Sample"]],
        slot_capacities: Sequence[int],
        fanout: int,
    ) -> None:
        self._buffer = buffer
        self._launch = launch
        self._pull = pull
        self._fanout = int(fanout)
        if self._fanout <= 0:
            raise ValueError(f"fanout must be positive; got {fanout}")
        self._budget = 0
        self._credit = 0
        self._inflight: Set[asyncio.Task] = set()
        self._pending: Set[Any] = set()
        self._resumed = asyncio.Event()
        self._resumed.set()
        self._idle = asyncio.Event()
        self._free_slots: asyncio.Queue = asyncio.Queue()
        for index, capacity in enumerate(slot_capacities):
            for _ in range(int(capacity)):
                self._free_slots.put_nowait(index)
        if self._free_slots.empty():
            raise ValueError("rollout producer needs at least one slot of capacity")
        self._task = asyncio.create_task(self._run(), name="rollout-producer")

    async def set_admission(self, *, max_outstanding: int, remaining_prompts: int) -> None:
        """Set the occupancy cap and recompute the admission credit; see the rollout README."""
        self._budget = max(0, int(max_outstanding))
        self._credit = max(0, int(remaining_prompts) - len(self._inflight) - len(self._buffer))

    @property
    def inflight(self) -> int:
        self._raise_if_dead()
        return len(self._inflight)

    async def _run(self) -> None:
        while True:
            if not self._resumed.is_set() and not self._inflight:
                self._idle.set()
                await self._resumed.wait()
                self._idle.clear()
            while (
                self._resumed.is_set() and self._credit > 0 and len(self._inflight) + len(self._buffer) < self._budget
            ):
                prompt = self._pull()
                if prompt is None:
                    break
                self._credit -= 1
                self._inflight.add(asyncio.create_task(self._run_group(prompt)))
            if not self._inflight:
                await asyncio.sleep(_IDLE_INTERVAL_S)
                continue
            # Sole owner of _inflight: nothing else may await it, or a group would be put twice.
            done, self._inflight = await asyncio.wait(self._inflight, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                await self._buffer.put(task.result())

    async def _run_group(self, prompt: "Sample") -> BufferEntry:
        samples = await asyncio.gather(*(self._launch_one(prompt) for _ in range(self._fanout)))
        return BufferEntry(prompt=prompt, group=list(samples))

    async def _launch_one(self, prompt: "Sample") -> "Sample":
        # One slot per sibling, released in finally: no task holds a slot while waiting for another.
        index = await self._free_slots.get()
        try:
            pending = self._launch(index, prompt)
            self._pending.add(pending)
            try:
                return await pending.aresult()
            finally:
                self._pending.discard(pending)
        finally:
            self._free_slots.put_nowait(index)

    async def next_group(self, current_version: int) -> Group:
        get = asyncio.create_task(self._buffer.get(current_version))
        try:
            while True:
                done, _ = await asyncio.wait(
                    {get, self._task}, return_when=asyncio.FIRST_COMPLETED, timeout=_NO_PROGRESS_WARN_S
                )
                # Checked before the get: a dead producer fails this step now, not after its backlog drains.
                if self._task in done:
                    self._task.result()
                    raise RuntimeError("rollout producer exited without an exception")
                if get in done:
                    return get.result()
                logger.warning("no completed rollout group for %ss", _NO_PROGRESS_WARN_S)
        finally:
            if not get.done():
                get.cancel()

    async def pause(self) -> None:
        """Stop admitting, then wait for the producer to settle every in-flight group into the buffer."""
        self._resumed.clear()
        idle = asyncio.create_task(self._idle.wait())
        try:
            done, _ = await asyncio.wait({idle, self._task}, return_when=asyncio.FIRST_COMPLETED)
            if self._task in done:
                self._task.result()
                raise RuntimeError("rollout producer exited without an exception")
        finally:
            if not idle.done():
                idle.cancel()

    async def resume(self) -> None:
        self._raise_if_dead()
        self._resumed.set()

    async def close(self) -> None:
        self._resumed.clear()
        # Captured before cancelling, because _launch_one drops each entry on the way out.
        outstanding = list(self._pending)
        self._task.cancel()
        for task in list(self._inflight):
            task.cancel()
        await asyncio.gather(self._task, *self._inflight, return_exceptions=True)
        self._inflight.clear()
        self._pending.clear()
        for pending in outstanding:
            pending.discard_on_completion()

    def _raise_if_dead(self) -> None:
        if self._task.done() and not self._task.cancelled():
            self._task.result()
            raise RuntimeError("rollout producer exited without an exception")


__all__ = ["Producer"]
