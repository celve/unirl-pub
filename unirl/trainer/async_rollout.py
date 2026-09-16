"""Shared driver-side policy for the async batch trainers (AR and diffusion)."""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import TYPE_CHECKING, Callable, Deque, Dict, List, Optional, Tuple

from unirl.rollout.manager import (
    RolloutManager,
    keep_within_lag,
    required_worker_concurrency,
    validate_worker_inflight,
    well_formed_group,
)
from unirl.trainer.base import unwrap_replicated_int
from unirl.types.sampling import total_samples_per_prompt

if TYPE_CHECKING:
    from unirl.types.sample import Sample

logger = logging.getLogger(__name__)


def resolve_separate_worker_concurrency(
    *,
    num_devices: int,
    train_fraction: float,
    per_worker_inflight: int,
    configured_concurrency: Optional[int],
    engine_concurrency: Optional[int],
) -> Tuple[int, int | List[int]]:
    """Resolve serial train workers and concurrent rollout workers before pool setup."""
    train_devices = round(train_fraction * num_devices)
    if train_devices <= 0 or train_devices >= num_devices:
        raise ValueError(
            f"train_fraction={train_fraction} yields {train_devices} train devices "
            f"of {num_devices}; must leave a non-empty rollout slab."
        )
    rollout_concurrency = (
        required_worker_concurrency(per_worker_inflight) if configured_concurrency is None else configured_concurrency
    )
    engine_concurrency = None if engine_concurrency is None else int(engine_concurrency)
    validate_worker_inflight(
        per_worker_inflight,
        worker_max_concurrency=rollout_concurrency,
        engine_concurrency=engine_concurrency,
    )
    worker_concurrency = (
        [1] * train_devices + [rollout_concurrency] * (num_devices - train_devices)
        if configured_concurrency is None
        else rollout_concurrency
    )
    return train_devices, worker_concurrency


def next_hard_boundary(
    trained_batches: int,
    *,
    num_rollouts: int,
    eval_interval: int = 0,
    save_interval: int = 0,
) -> int:
    boundary = num_rollouts
    for interval in (eval_interval, save_interval):
        if interval > 0 and trained_batches < num_rollouts:
            boundary = min(boundary, ((trained_batches // interval) + 1) * interval)
    return boundary


def boundary_admission(
    *,
    max_inflight_prompts: int,
    batch_size: int,
    trained_batches: int,
    hard_boundary: int,
) -> Tuple[int, int]:
    """Occupancy cap, and the prompts still admissible before the next eval/save/final boundary."""
    if max_inflight_prompts % batch_size:
        raise ValueError(
            f"max_inflight_prompts ({max_inflight_prompts}) must be divisible by batch_size ({batch_size})"
        )
    return max_inflight_prompts, max(0, hard_boundary - trained_batches) * batch_size


def rollout_version_metrics(
    *,
    train_version: int,
    output_version: int,
    num_updates_per_batch: int,
    version_spread: int = 0,
) -> dict[str, float]:
    staleness = train_version - output_version
    if staleness < 0:
        raise ValueError(f"rollout batch has future output version {output_version} > train version {train_version}")
    return {
        "async/output_version": output_version,
        "async/staleness_updates": staleness,
        "async/staleness_batches": staleness / num_updates_per_batch,
        "async/version_spread": version_spread,
    }


def training_version_metrics(
    *,
    train_version: int,
    published_version: int,
    optimizer_updates: int,
    batches_since_sync: int,
) -> dict[str, int]:
    return {
        "async/train_version": train_version,
        "async/published_version": published_version,
        "async/publish_lag": train_version - published_version,
        "async/optimizer_updates": optimizer_updates,
        "async/batches_since_sync": batches_since_sync,
    }


def combine_rollout_prompts(
    groups: List[List["Sample"]],
    *,
    require_single_rollout_id: bool = False,
) -> Tuple["Sample", int, int]:
    """Combine completed prompt trees, attributing the batch to its oldest behavior policy; see rollout README."""
    chunks = [sample for group in groups for sample in group]
    if not chunks:
        raise ValueError("cannot combine an empty rollout result")
    if require_single_rollout_id:
        rollout_ids = {_rollout_id(sample) for sample in chunks}
        if len(rollout_ids) != 1:
            raise RuntimeError(
                "rollout batch combines multiple generation ids with incompatible shared schedules: "
                f"{sorted(rollout_ids)}"
            )
    versions = {part.output_version for sample in chunks for part in sample.gen_parts()}
    if not versions or None in versions:
        raise RuntimeError("rollout batch is missing output_version provenance")
    if require_single_rollout_id and len(versions) != 1:
        raise RuntimeError(f"rollout batch has mixed output versions: {sorted(versions)}")
    output_version = min(versions)
    version_spread = max(versions) - output_version
    if len(chunks) == 1:
        return chunks[0], output_version, version_spread

    from unirl.types.sample import Sample

    return _restamp_output_version(Sample.concat(chunks), output_version), output_version, version_spread


def _restamp_output_version(sample: "Sample", output_version: int) -> "Sample":
    """Overwrite every gen Part's version, which ``Batch.concat`` resolved to the first chunk's."""
    parts = [part.fill(output_version=output_version) if part.is_gen else part for part in sample.parts]
    return sample.with_parts(parts)


def _rollout_id(sample: "Sample") -> int:
    if not sample.parts or not sample.parts[0].metadata:
        raise RuntimeError("rollout Sample has no root rollout_id metadata")
    values = {row.get("rollout_id") for row in sample.parts[0].metadata}
    if None in values or len(values) != 1:
        raise RuntimeError(f"rollout Sample must carry one root rollout_id; got {values}")
    return int(next(iter(values)))


class AsyncRolloutTrainerMixin:
    """One async batch loop shared by ``AsyncARTrainer`` and ``AsyncDiffusionTrainer``."""

    _require_single_generation = False

    def _async_wandb_extra(self) -> Dict[str, object]:
        """Trainer-specific keys merged into the wandb run config."""
        return {}

    def _refill_before_score(self) -> bool:
        """Whether replacement rollout work may start before reward scoring."""
        return False

    def _boundary_evaluate(self, rollout_id: int, *, initial: bool) -> None:
        """Run the trainer's evaluation at a synced, empty rollout boundary."""
        raise NotImplementedError

    def _make_prompt_source(self) -> Tuple[Callable[[], Optional["Sample"]], Callable[["Sample"], None]]:
        """A pull/recycle pair over one deque, both called only from the manager's loop thread."""
        ready: Deque["Sample"] = deque()

        def pull() -> Optional["Sample"]:
            if not ready:
                request = self._build_request_sample(
                    self.data_source.get_samples(self.batch_size),
                    self._next_generation_id,
                )
                self._next_generation_id += 1
                prompts = request.split()
                if len(prompts) != self.batch_size:
                    raise RuntimeError(f"request batch split into {len(prompts)} prompts; expected {self.batch_size}")
                ready.extend(prompts)
            return ready.popleft() if ready else None

        return pull, ready.append

    def _score_completed(self, rollout_id: int, completed: "Sample") -> "Sample":
        scored = self.reward.score_and_attach(completed)
        self._drop_decoded(scored, rollout_id=rollout_id)
        return scored

    def _train_async_loop(
        self,
        *,
        num_rollouts: int,
        save_interval: int,
        save_dir: Optional[str],
        load_dir: Optional[str],
        save_mode: str,
    ) -> None:
        start_rollout = self.maybe_load_checkpoint(load_dir, num_rollouts=num_rollouts)
        resumed = bool(load_dir)
        self._train_version = unwrap_replicated_int(
            self.backend.get_optimizer_step_count(),
            name="backend optimizer step count",
        )
        self._batches_since_sync = 0
        for _ in range(start_rollout):
            self.data_source.get_samples(self.batch_size)
        staleness_budget = self._max_staleness * self._num_updates_per_batch
        self._init_wandb(
            num_rollouts=num_rollouts,
            extra={
                "max_inflight": self._max_inflight,
                "max_inflight_prompts": self._max_inflight_prompts,
                "per_worker_inflight": self._per_worker_inflight,
                "weight_sync_interval": self._weight_sync_interval,
                "max_staleness": self._max_staleness,
                "staleness_budget": staleness_budget,
                "num_updates_per_batch": self._num_updates_per_batch,
                **self._async_wandb_extra(),
            },
        )

        self._next_generation_id = start_rollout
        engine_slots = self.rollout.engine_slots
        pull, recycle = self._make_prompt_source()
        self._rollout_manager = RolloutManager(
            self.rollout,
            launch=lambda index, sample: engine_slots[index].launch("generate_on_slot", sample),
            pull=pull,
            recycle=recycle,
            put_filter=well_formed_group(total_samples_per_prompt(self.sampling_params)),
            get_filter=keep_within_lag(staleness_budget),
            slot_capacities=[self._per_worker_inflight] * len(engine_slots),
            fanout=1,
        )

        if resumed or self.eval_interval > 0:
            self._sync_rollout(force=True, require_empty=True)
        if self.eval_interval > 0:
            self._boundary_evaluate(start_rollout, initial=True)

        try:
            for rollout_id in range(start_rollout, num_rollouts):
                t0 = time.perf_counter()
                hard_boundary = next_hard_boundary(
                    rollout_id,
                    num_rollouts=num_rollouts,
                    eval_interval=self.eval_interval,
                    save_interval=save_interval,
                )
                sample, output_version, version_spread = self._next_rollout_batch(
                    rollout_id,
                    hard_boundary=hard_boundary,
                )
                training_progress = rollout_id / max(1, num_rollouts - 1)
                result, mean_reward = self._advantage_and_train(
                    sample,
                    training_progress=training_progress,
                    rollout_id=rollout_id,
                    t0=t0,
                    extra_metrics={
                        **rollout_version_metrics(
                            train_version=self._train_version,
                            output_version=output_version,
                            num_updates_per_batch=self._num_updates_per_batch,
                            version_spread=version_spread,
                        ),
                        **self._rollout_manager.drain_metrics(),
                    },
                )
                self.wandb_logger.log_progress(rollout_id, num_rollouts, result, mean_reward, logger=logger)

                step = rollout_id + 1
                eval_due = self.eval_interval > 0 and step % self.eval_interval == 0
                save_due = save_interval > 0 and (step % save_interval == 0 or step >= num_rollouts)
                sync_due = step < num_rollouts and self._batches_since_sync >= self._weight_sync_interval
                if eval_due or save_due or sync_due:
                    self._sync_rollout(require_empty=eval_due or save_due)

                if step >= num_rollouts and not self._rollout_manager.empty:
                    raise RuntimeError("final rollout boundary requires an empty RolloutManager")

                if eval_due:
                    self._boundary_evaluate(rollout_id, initial=False)
                if save_due:
                    self.maybe_save_checkpoint(
                        rollout_id,
                        num_rollouts,
                        save_interval=save_interval,
                        save_dir=save_dir,
                        save_mode=save_mode,
                    )
        finally:
            try:
                self._rollout_manager.close()
            finally:
                self._finish_wandb()

    def _sync_rollout(self, *, force: bool = False, require_empty: bool = False) -> None:
        manager = self._rollout_manager
        needs_publish = force or manager.published_version != self._train_version
        if not needs_publish:
            if require_empty and not manager.empty:
                raise RuntimeError("eval/checkpoint boundary requires an empty RolloutManager")
            self._batches_since_sync = 0
            return

        # Completed groups stay buffered across the publication; the lag filter decides
        # at consume time whether they are still trainable.
        manager.publish(self.weight_sync, output_version=self._train_version)
        if require_empty and not manager.empty:
            raise RuntimeError("eval/checkpoint boundary requires an empty RolloutManager")
        self._batches_since_sync = 0

    def _next_rollout_batch(
        self,
        rollout_id: int,
        *,
        hard_boundary: int,
    ) -> Tuple["Sample", int, int]:
        self._set_admission(trained_batches=rollout_id, hard_boundary=hard_boundary)

        manager = self._rollout_manager
        groups = [manager.next_group(current_version=self._train_version) for _ in range(self.batch_size)]
        completed, output_version, version_spread = combine_rollout_prompts(
            groups,
            require_single_rollout_id=self._require_single_generation,
        )

        scored = None
        if not self._refill_before_score():
            scored = self._score_completed(rollout_id, completed)

        # Consuming a batch releases capacity immediately. Refill before reward
        # for AR and before training for trainers that require reap-time scoring.
        self._set_admission(trained_batches=rollout_id + 1, hard_boundary=hard_boundary)

        if scored is None:
            scored = self._score_completed(rollout_id, completed)

        return scored, output_version, version_spread

    def _set_admission(self, *, trained_batches: int, hard_boundary: int) -> None:
        max_outstanding, remaining_prompts = boundary_admission(
            max_inflight_prompts=self._max_inflight_prompts,
            batch_size=self.batch_size,
            trained_batches=trained_batches,
            hard_boundary=hard_boundary,
        )
        self._rollout_manager.set_admission(max_outstanding=max_outstanding, remaining_prompts=remaining_prompts)


__all__ = [
    "AsyncRolloutTrainerMixin",
    "boundary_admission",
    "combine_rollout_prompts",
    "next_hard_boundary",
    "resolve_separate_worker_concurrency",
    "rollout_version_metrics",
    "training_version_metrics",
]
