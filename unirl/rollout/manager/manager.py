"""Driver-side rollout manager: a producer loop behind a synchronous trainer API; see :class:`RolloutManager`."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, Tuple

from unirl.rollout.manager.buffer import GroupBuffer
from unirl.rollout.manager.filters import GetFilter, Group, PutFilter
from unirl.rollout.manager.producer import Producer

if TYPE_CHECKING:
    from unirl.distributed.group.handle import Handle
    from unirl.types.sample import Sample

logger = logging.getLogger(__name__)

_WORKER_CONTROL_CONCURRENCY = 2


class _LoopThread:
    """One event loop on a named daemon thread, which the trainer bridges into synchronously."""

    def __init__(self, label: str) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name=label, daemon=True)
        self._thread.start()

    def run(self, coro) -> Any:
        if threading.current_thread() is self._thread:
            raise RuntimeError("blocking on the manager loop thread would deadlock it")
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result()

    def close(self) -> None:
        # Stop, never cancel: the loop must finish its callbacks before the thread joins.
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._loop.close()


class RolloutManager:
    """Produces scored prompt groups continuously on its own loop; the trainer pulls one at a time."""

    def __init__(
        self,
        rollout: "Handle",
        *,
        launch: Callable[[int, "Sample"], Any],
        pull: Callable[[], Optional["Sample"]],
        put_filter: PutFilter,
        get_filter: GetFilter,
        slot_capacities: Sequence[int],
        fanout: int,
        recycle: Optional[Callable[[Any], None]] = None,
    ) -> None:
        self._rollout = rollout
        self._published_version = 0
        self._closed = False
        self._loop = _LoopThread("rollout-manager")
        self._buffer, self._producer = self._loop.run(
            self._build(
                launch=launch,
                pull=pull,
                put_filter=put_filter,
                get_filter=get_filter,
                slot_capacities=slot_capacities,
                fanout=fanout,
                recycle=recycle,
            )
        )

    async def _build(self, **kwargs) -> Tuple[GroupBuffer, Producer]:
        """Construct on the loop thread so every Condition, Event and Queue binds to the loop that awaits it."""
        buffer = GroupBuffer(
            put_filter=kwargs["put_filter"],
            get_filter=kwargs["get_filter"],
            recycle=kwargs["recycle"],
        )
        producer = Producer(
            buffer,
            launch=kwargs["launch"],
            pull=kwargs["pull"],
            slot_capacities=kwargs["slot_capacities"],
            fanout=kwargs["fanout"],
        )
        return buffer, producer

    def next_group(self, *, current_version: int) -> Group:
        self._ensure_open()
        current_version = int(current_version)
        if current_version < 0:
            raise ValueError(f"current_version must be non-negative; got {current_version}")
        return self._loop.run(self._producer.next_group(current_version))

    def publish(self, weight_sync: object, *, output_version: int) -> int:
        """Settle in-flight generation, push weights from the trainer thread, then resume producing."""
        self._ensure_open()
        next_version = int(output_version)
        if next_version < self._published_version:
            raise ValueError(
                f"output_version must be monotonic; current={self._published_version}, next={next_version}"
            )
        self._loop.run(self._producer.pause())
        # Deliberately not in a try/finally: a failed weight write must leave the producer paused
        # rather than resume generation against half-published weights.
        weight_sync.sync()
        self._rollout.set_version(next_version)
        self._published_version = next_version
        self._loop.run(self._producer.resume())
        return self._published_version

    def pause(self) -> None:
        self._ensure_open()
        self._loop.run(self._producer.pause())

    def resume(self) -> None:
        self._ensure_open()
        self._loop.run(self._producer.resume())

    def set_admission(self, *, max_outstanding: int, remaining_prompts: int) -> None:
        """Cap concurrent work and spend down the prompts still admissible before the next hard boundary."""
        self._ensure_open()
        self._loop.run(
            self._producer.set_admission(max_outstanding=max_outstanding, remaining_prompts=remaining_prompts)
        )

    def drain_metrics(self) -> dict[str, int]:
        return {**self._buffer.drain_metrics(), "rollout/buffered_groups": len(self._buffer)}

    @property
    def published_version(self) -> int:
        return self._published_version

    @property
    def counts(self) -> Tuple[int, int]:
        """In-flight and buffered group counts, accurate on the trainer thread at a paused boundary."""
        return self._producer.inflight, len(self._buffer)

    @property
    def empty(self) -> bool:
        return not self._producer.inflight and not len(self._buffer)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._loop.run(self._producer.close())
        except BaseException:
            # Teardown path: log instead of re-raising so close() cannot mask the primary error.
            logger.exception("RolloutManager.close: discarding in-flight rollout work")
        finally:
            self._loop.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("RolloutManager is closed")


def required_worker_concurrency(per_worker_inflight: int) -> int:
    """Return actor concurrency that leaves room for rollout control calls."""
    return per_worker_inflight + _WORKER_CONTROL_CONCURRENCY


def validate_worker_inflight(
    per_worker_inflight: int,
    *,
    worker_max_concurrency: int,
    engine_concurrency: Optional[int],
) -> None:
    """Validate per-worker rollout concurrency against actor and engine capacity."""
    if per_worker_inflight <= 0:
        raise ValueError(f"per_worker_inflight must be positive; got {per_worker_inflight}")

    # Reserve control-call threads for sleep and weight sync while rollout calls occupy their own slots.
    required_concurrency = required_worker_concurrency(per_worker_inflight)
    if worker_max_concurrency < required_concurrency:
        raise ValueError(
            f"worker_max_concurrency ({worker_max_concurrency}) must be >= per_worker_inflight "
            f"+ {_WORKER_CONTROL_CONCURRENCY} "
            f"({required_concurrency}) so rollout calls cannot starve control calls"
        )

    if engine_concurrency is not None and per_worker_inflight > engine_concurrency:
        raise ValueError(
            f"per_worker_inflight ({per_worker_inflight}) exceeds engine concurrency ({engine_concurrency})"
        )


__all__ = ["RolloutManager", "required_worker_concurrency", "validate_worker_inflight"]
