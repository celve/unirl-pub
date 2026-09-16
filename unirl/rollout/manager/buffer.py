"""FIFO of finished rollout groups with a rejection point on each side; see the rollout README."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from unirl.rollout.manager.filters import Disposition, GetFilter, Group, PutFilter


@dataclass
class BufferEntry:
    """A finished group beside the prompt it came from, so recycling reconstructs nothing."""

    prompt: Any
    group: Group


class GroupBuffer:
    """Group FIFO whose depth the producer bounds; ``get`` waits for work, ``put`` never blocks."""

    def __init__(
        self,
        *,
        put_filter: PutFilter,
        get_filter: GetFilter,
        recycle: Optional[Callable[[Any], None]] = None,
    ) -> None:
        self._put_filter = put_filter
        self._get_filter = get_filter
        self._recycle = recycle
        self._entries: List[BufferEntry] = []
        self._condition = asyncio.Condition()
        self._accepted = 0
        self._dropped = 0
        self._recycled = 0

    async def put(self, entry: BufferEntry) -> None:
        verdict = self._put_filter(entry.group)
        if verdict is not Disposition.KEEP:
            # Rejected before the lock, so a reject never contends with a waiting consumer.
            self._dispose(verdict, entry)
            return
        async with self._condition:
            self._entries.append(entry)
            self._condition.notify_all()

    async def get(self, current_version: int) -> Group:
        async with self._condition:
            while True:
                while not self._entries:
                    await self._condition.wait()
                entry = self._entries.pop(0)
                # Nothing awaits between the pop and the return, so a cancelled get cannot lose a
                # group and an acceptance is booked in the same turn the entry leaves the buffer.
                if self._get_filter(entry.group, current_version) is Disposition.KEEP:
                    self._accepted += 1
                    return entry.group
                self._dispose(Disposition.RECYCLE, entry)

    @property
    def accepted(self) -> int:
        """Groups delivered since the last :meth:`reset_accepted`, counted atomically with the pop."""
        return self._accepted

    def reset_accepted(self) -> None:
        self._accepted = 0

    def _dispose(self, verdict: Disposition, entry: BufferEntry) -> None:
        if verdict is Disposition.RECYCLE and self._recycle is not None:
            self._recycled += 1
            self._recycle(entry.prompt)
            return
        self._dropped += 1

    def drain_metrics(self) -> dict[str, int]:
        """Rejection counts since the last read, which resets the window."""
        metrics = {"rollout/dropped_groups": self._dropped, "rollout/recycled_groups": self._recycled}
        self._dropped = 0
        self._recycled = 0
        return metrics

    def __len__(self) -> int:
        return len(self._entries)


__all__ = ["BufferEntry", "GroupBuffer"]
