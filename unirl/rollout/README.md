# Rollout

> **Where it fits:** the *rollout* step of the loop —
> **rollout** → reward → advantage → train → sync. In: a request `Sample` from
> the trainer. Out: a filled `Sample`. Full map: [`../README.md`](../README.md).

<div align="center">
  <img src="../../assets/rollout-engines-new.png" alt="UniRL rollout engines selected by _target_ across direct, separate, and colocated deployment modes" width="100%">
</div>

*Every rollout engine fills one `Sample`. Agentic scheduling and group assembly live
in the driver-side rollout manager.*

## What it is

`unirl.rollout` owns the rollout engines — the box that fills a typed `Sample` by
running a model pipeline and the SDE step kernels. The agentic engine runs the
model/environment turns for one trajectory. It does not compute reward or loss.

## Why it exists

The rollout can come from two unrelated codebases — the in-process training
`Pipeline`, or the SGLang fork sampling in its own subprocess. For on-policy RL they
must walk a *numerically identical* trajectory, because the trainer replays the
rollout to recompute log-probs and any drift silently pushes the GRPO ratio off 1.0.
So this module is a **verification boundary**, not just a backend-hiding shim: it
pins one σ schedule on the generated `Part`'s sampling params and verifies what the
backend used (`engine/sigma_verify.py`). Each engine adapts its backend wire format
into canonical `Part` fields, so a dedicated server swaps in for the training model
without the loop noticing, and a mismatch crashes loudly instead of training on a
wrong objective.

## How it works

- **One generation interface.** `BaseRolloutEngine` (`engine/base.py`)
  is a `Remote` whose concrete engines implement synchronous `generate(sample)`;
  each returns one `Sample`. Batch engines dispatch `generate` with `DP_SCATTER`.
  Agentic `generate` is undecorated because the driver manager addresses one
  engine slot per trajectory; Ray actor concurrency lets the inner backend batch
  concurrent calls.
- **The typed boundary** (`../types/`). A `Sample` is an ordered chain of `Part`s.
  Each Part carries lineage ids, a raw `primitive`, an encoded `segment`, replay
  conditions, sampling params (including the σ schedule), and optional decoded
  media. Single-stage flows fill one generated Part; composed PE fills its chained
  AR and diffusion Parts.
- **The engines.** `trainside` (in-process — the train actor's pipeline *is* the
  sampler), `sglang_diffusion` (dedicated diffusion), `sglang` (dedicated AR), `vllm_omni`
  (dedicated; BAGEL / HI3 / SD3 / HunyuanVideo), `fastvideo` (dedicated accelerated video
  sampling), and `composed` (chains an AR child + a
  diffusion child for prompt enhancement) are the six single-turn engines.
  `agentic` wraps one of them with an environment to produce multi-turn
  trajectories. Each diffusion engine consumes the Part's pinned sigmas verbatim
  and reads the same driver-authored `NoiseRecipe` (`../types/noise_recipe.py`),
  but realizes it differently: `trainside`, `sglang_diffusion` and `vllm_omni`
  resolve the recipe to an `x_T` tensor, so those three start a rollout from
  bit-identical noise; `fastvideo` cannot accept a tensor and instead derives
  per-sample seeds from the recipe's noise-group ids, so its noise matches only
  in grouping, not bit-for-bit. `forward_batch_size` bounds peak memory by
  slicing the Sample and concatenating the results.
- **Deployment modes:** *direct sampling* — the trainside engine, no `sync:`, the
  ratio is 1 on the first update; *separate* — a dedicated engine on its own GPUs
  plus a `sync:` block; *colocate* — a dedicated engine sharing GPUs with train,
  plus offload/onload and `sync:`.
- **Driver-side scheduling.** One `manager.RolloutManager` serves batch and agentic
  trainers. It owns an asyncio loop on its own daemon thread: a producer coroutine
  keeps work outstanding, one task per prompt group with siblings gathered inside it,
  and finished groups land in a FIFO the trainer pulls one at a time. Admission is an
  occupancy cap plus a delivery horizon the trainer recomputes each batch, so the
  manager drains empty at an eval, save or final boundary. The manager owns the
  published rollout version; trainers own training progress, publication cadence and
  scoring order. `AgenticTrainer` drives the same manager as a per-step barrier.
  Publication and durable boundaries still settle in-flight work before proceeding.

**Extending it:** a new single-turn engine adds `engine/<name>/config.py` (a
`BaseEngineConfig` whose `make_engine(**deps)` lazily imports and builds it) and
`engine/<name>/engine.py` (subclass `BaseRolloutEngine`, implement
synchronous generation over the whole-`Sample` contract — thread-safe for
concurrent callers if it should serve as an agentic inner, else serialized
internally — and dispatch `generate` with `DP_SCATTER`). A dedicated engine also
implements its weight-receive method and a matching `sync:` handler in
`../distributed/weight_sync`.

## Engine anatomy, and adding a model to an existing engine

Engine dirs use two layouts. The compact engines (`trainside`, `fastvideo`,
`composed`, `agentic`) contain `config.py` + `engine.py`. Server-backed engines
(`sglang`, `sglang_diffusion`, `vllm_omni`) also carry `adapters/`
(per-family/modality wire-format translation), `backends/` (server process
management), `utils/`, `weight_sync.py`, and a
runtime-patch dir for the pinned upstream (`sglang_diffusion/_patches/`,
`vllm_omni/patches/`). `vllm_omni` additionally carries worker-subprocess code
(`pipelines/`, `worker/`) and stage boot configs (`stage_configs/`).

Model onboarding is per-engine, and the adapter file is usually **not** the whole
change surface:

- **`sglang` (AR/VLM):** onboarding is normally config-only: text models use the
  `text` adapter, while `image_token` selects `vlm`. Add and register a new adapter
  only for a genuinely new wire shape, extending `TextLMAdapter` or `VLMAdapter`
  and importing it in `adapters/__init__.py`.
- **`sglang_diffusion`:** add `adapters/<family>.py` extending `ImageAdapter` or
  `VideoAdapter`, register it by `model_family`, and import it in
  `adapters/__init__.py`. New condition fields that cross the wire also need an
  entry in `_COND_FIELDS` and either `_POS_MAP` / `_NEG_MAP` or an explicit copy
  branch in `_copy_conditions`; add `_patches/hijack.py` wiring only when the
  model needs a new upstream patch.
- **`vllm_omni`:** add an `adapters/<family>.py` binder (keyed by modality),
  register it, import it in `adapters/__init__.py`, and add the appropriate boot
  YAML under `stage_configs/`. DiT families additionally need a worker-side
  `pipelines/<model>/pipeline.py`; if the AR/DiT worker needs new behavior, add a
  `worker/` extension or `patches/compat_<model>.py`.

## Gotchas

- **Never recompute σ inside an engine** — the generated Part's pinned sigmas are
  the single source of truth; `engine/sigma_verify.py` checks the backend echo (it
  guards the GRPO log-prob ratio).
- **Batch `generate` must dispatch `DP_SCATTER`.** Agentic is the intentional
  exception: its undecorated method is reached through one `Handle.slot(...)`.
- **Direct sampling forbids a `sync:` block; dedicated requires one.** The trainside
  engine also can't live on a `layout: separate` slab — `_build_rollout` raises.
- **`RolloutManager.publish()` is the only weight-publication path on async trainers** —
  it pauses the producer, settles in-flight groups into the buffer, pushes weights, and
  resumes. It deliberately has no `try/finally`: a failed weight write leaves the
  producer paused rather than resuming against half-published weights. Buffered groups
  survive a publication and stay subject to the get-side filter, so publication no
  longer batch-aligns anything. Eval/checkpoint boundaries still require an empty
  manager. Scoring order remains trainer policy.
- **`weight_sync.sync()` and `set_version()` run on the trainer thread, never the loop
  thread** — the grad context is a `threading.local`, so a train-slab `@distributed`
  call made from the manager's loop thread silently loses autograd instead of raising.
  The loop thread owns the rollout Handle; the trainer thread owns the train slab.
- **A dead producer surfaces on the next `next_group`, not after its backlog drains** —
  the consumer races the buffer `get` against the producer task and checks the task
  first, so a launcher or engine failure fails the step immediately. That race is the
  whole liveness contract; there is no separate failure flag.
- **`GroupBuffer.get` must not await between the pop and the return** — a cancelled
  `get` would lose the popped group. Keep the get-side filter and the recycle callback
  synchronous.
- **Backpressure is the producer's admission, not a blocking `put`** — a blocking `put`
  would deadlock `pause()`, because the drain cannot complete while the producer waits
  for buffer space no one is consuming. Admission has two bounds and they are not
  interchangeable: an **occupancy cap** (`max_outstanding`) bounding concurrent work, and a
  **delivery horizon** (`remaining_prompts`) bounding how many groups may still be *accepted*
  before the next boundary. The producer admits while `outstanding < cap` and
  `outstanding + accepted < horizon`. Only the horizon keeps the manager empty at a boundary —
  an occupancy target alone is refilled by every `get`, so the producer admits straight past it.
  The horizon must count **accepted deliveries, not pulls**: a group rejected on get is recycled
  and never delivered, so spending its allowance would strand the recycled prompt with nothing
  left to re-admit it and hang the consumer inside `next_group`.
- **Only the producer coroutine may await `_inflight`** — `pause()` waits on an idle
  event the producer sets, because two coroutines awaiting the same task set would each
  put the same finished group.
- **`output_version` is attributed to where generation started, not where it ended** —
  engines capture the version before calling the backend, so a publication landing
  mid-generation over-reports staleness by at most the publications it spanned rather
  than silently claiming the work is fresher than it is. Filters therefore discard
  more than strictly necessary and never keep off-policy work as on-policy.
- **A mixed-version batch is attributed to its oldest span, and re-stamped explicitly** —
  `output_version` is a `shared_field`, so `Batch.concat` resolves it to the first
  chunk's value; `combine_rollout_prompts` overwrites every gen Part with
  `min(versions)` afterwards, or the batch would merely *look* single-version to every
  later reader. Diffusion keeps the single-version rejection, because `sampling_params`
  is shared the same way and carries the pinned σ/SDE schedule that
  `engine/sigma_verify.py` guards; `async/version_spread` must be zero there.
- **Request-level partial rollout is gone** — nothing arms `set_stopping`, so an
  agentic trajectory always runs to terminal completion and `harness_status ==
  "suspended"` is unreachable. The engine-side suspension contract is left intact for
  a future engine-level retract.
- **A resolve/route failure poisons the `RolloutManager`** — samples may already be
  lost, so every later call (including `empty` / `counts`) re-raises the original
  error rather than reporting clean state; only `close()` stays safe.
- **Reward/advantage methods are not engine code** — `Part.compute_advantages` and
  `Sample.propagate_rewards` are called by the trainer after scoring. An engine
  fills generation fields such as `segment`, `conditions`, `primitive`, and
  `media_preview`; rewards arrive later from `RewardService`.
