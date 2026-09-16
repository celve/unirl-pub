"""Group dispositions and the filters the rollout manager applies on put and get."""

from __future__ import annotations

from collections import Counter
from enum import Enum
from typing import TYPE_CHECKING, Callable, List

if TYPE_CHECKING:
    from unirl.types.sample import Sample

Group = List["Sample"]
PutFilter = Callable[[Group], "Disposition"]
GetFilter = Callable[[Group, int], "Disposition"]


class Disposition(Enum):
    """What the buffer does with a group a filter rejected."""

    KEEP = "keep"
    DROP = "drop"
    RECYCLE = "recycle"


def roots_of(sample: "Sample") -> List[str]:
    """Ordered unique root ids of ``sample``; raises when it has none."""
    if not sample.parts:
        raise ValueError("rollout Sample has no Parts")
    roots = list(dict.fromkeys(sample.root_group_ids(0)))
    if not roots:
        raise ValueError("rollout Sample has no root ids")
    return roots


def well_formed_group(group_size: int) -> PutFilter:
    """Put-side: one root prompt fanned out to exactly ``group_size`` stamped descendants."""
    group_size = int(group_size)
    if group_size <= 0:
        raise ValueError(f"group_size must be positive; got {group_size}")

    def judge(group: Group) -> Disposition:
        if not group:
            raise RuntimeError("rollout group is empty")
        roots = {root for sample in group for root in roots_of(sample)}
        if len(roots) != 1:
            raise RuntimeError(f"rollout group must carry exactly one root; got {sorted(roots)}")
        for sample in group:
            # A failed agentic trajectory legitimately carries no gen Parts; the trainer drops it at scoring.
            if sample.parts[-1].harness_status is None and not sample.gen_parts():
                raise RuntimeError("completed rollout has no generated Parts")
            unstamped = [index for index, part in enumerate(sample.gen_parts()) if part.output_version is None]
            if unstamped:
                raise RuntimeError(f"completed rollout has unstamped generated Parts at indices {unstamped}")
        descendants = Counter(root for sample in group for root in sample.root_group_ids(-1))
        if descendants != Counter({next(iter(roots)): group_size}):
            raise RuntimeError(f"rollout group fan-out does not match group_size={group_size}: {dict(descendants)}")
        return Disposition.KEEP

    return judge


def keep_within_lag(max_lag: int) -> GetFilter:
    """Get-side: judge a group by its oldest span and recycle it once that falls outside ``max_lag``."""
    max_lag = int(max_lag)
    if max_lag < 0:
        raise ValueError(f"max_lag must be non-negative; got {max_lag}")

    def judge(group: Group, current_version: int) -> Disposition:
        versions = [
            int(part.output_version)
            for sample in group
            for part in sample.gen_parts()
            if part.output_version is not None
        ]
        if not versions:
            return Disposition.KEEP
        if max(versions) > current_version:
            raise RuntimeError(f"rollout group has a future output version: {max(versions)} > {current_version}")
        return Disposition.KEEP if current_version - min(versions) <= max_lag else Disposition.RECYCLE

    return judge


def keep_regardless(group: Group, current_version: int) -> Disposition:
    """Get-side no-op, for barrier consumers whose groups cannot be stale."""
    del group, current_version
    return Disposition.KEEP


__all__ = [
    "Disposition",
    "GetFilter",
    "Group",
    "PutFilter",
    "keep_regardless",
    "keep_within_lag",
    "roots_of",
    "well_formed_group",
]
