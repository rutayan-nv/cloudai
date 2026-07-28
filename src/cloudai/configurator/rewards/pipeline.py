# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Composable reward pipeline.

Owned by the agent (specifically :class:`RLAgentBase`). State is in-memory
only; the pipeline does not read from or write to the env's reporting plane
(``trajectory.csv`` / ``env.csv``). Cold start on every fresh agent process.

Also exposes :func:`build_default_pipeline`, a pure function that produces
the framework-default transform stack from a tuple of context keys. Pulled
out as a free function (rather than a method) so it can be invoked from
non-agent contexts (offline analysis, tests, future tooling).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .transforms import (
    GlobalMeanStdFilter,
    PerContextZScore,
    RewardTransform,
)


class RewardPipeline:
    """Composes zero or more :class:`RewardTransform` instances in order.

    Per env step the pipeline:

    1. Calls ``update(reward, ctx)`` on every transform (in order).
    2. Threads the reward through each transform's ``apply(...)`` (in order)
       to produce the final transformed value passed to the agent.

    An empty pipeline is the identity. Subclasses are not required; behavior
    is configured by the list of transforms.
    """

    def __init__(self, transforms: Sequence[RewardTransform] | None = None) -> None:
        self._transforms: list[RewardTransform] = list(transforms or ())

    @property
    def transforms(self) -> tuple[RewardTransform, ...]:
        return tuple(self._transforms)

    def step(self, reward: float, ctx: Mapping[str, Any]) -> float:
        """Run one update + apply pass over all transforms.

        Semantics are **chained**: per transform, in pipeline order, the
        transform first ingests the value coming from upstream via
        :meth:`update`, then produces the value that flows downstream via
        :meth:`apply`. Concretely::

            r_0 = reward
            for t in transforms:
                t.update(r_i, ctx)
                r_{i+1} = t.apply(r_i, ctx)

        Each transform therefore sees the previous transform's output, not
        the original raw reward. This is the only consistent semantics when
        transforms are statistical (e.g. a per-context z-score that follows
        a global filter must track stats of the globally-normalized reward,
        not the raw reward).

        ``ctx`` is the **raw** observation/env-param dict (never the encoded
        policy input) and is forwarded unchanged to every transform.
        """
        out = float(reward)
        for transform in self._transforms:
            transform.update(out, ctx)
            out = float(transform.apply(out, ctx))
        return out

    def batch_step(
        self,
        rewards: Sequence[float],
        ctxs: Sequence[Mapping[str, Any]],
    ) -> list[float]:
        """Batch-mode entry point with the same semantics as ``step``.

        This is the API-level hook for parallel rollouts: callers (e.g., a
        future RLlib learner-side connector that gathers samples from
        multiple parallel env runners) build up ``(rewards, ctxs)`` lists,
        then hand them to the pipeline as a single call.

        The current implementation is a sequential loop over :meth:`step`
        — there is no parallel runner today (``num_env_runners=0``). When
        that changes, this method becomes the single place to swap in a
        parallel-friendly fold (e.g., compute per-bin batch stats in a
        parallel reduce, then update running stats once per batch).

        Raises
        ------
        ValueError
            If ``rewards`` and ``ctxs`` differ in length.
        """
        if len(rewards) != len(ctxs):
            raise ValueError(
                f"rewards / ctxs length mismatch: "
                f"len(rewards)={len(rewards)} vs len(ctxs)={len(ctxs)}"
            )
        return [self.step(r, c) for r, c in zip(rewards, ctxs)]


def build_default_pipeline(context_keys: tuple[str, ...]) -> RewardPipeline:
    """Framework-default transform stack: exactly **one** estimator.

    The estimator is selected from the context structure (Design X), never
    chained:

    * ``context_keys`` non-empty (the detector found a low-cardinality
      categorical context) → a single :class:`PerContextZScore` applied to the
      **raw** reward. It centers/scales each regime against its own running
      stats, removing the between-regime bias the value head would otherwise
      learn.
    * ``context_keys`` empty → a single :class:`GlobalMeanStdFilter`. With no
      bins to condition on, a regime-blind global z-score is the right (and
      only) estimator.

    Why not chain ``GlobalMeanStdFilter`` in front of ``PerContextZScore``:
    under EMA the global stage amplifies the regime gap (÷ a small running σ)
    and feeds warm-up spikes into the per-context stats, poisoning them for the
    rest of the run (see ``domain_randomization/scripts/viz_zscore_chain.py``).
    The two are *alternatives* keyed on context structure, not a pipeline.

    Pure function: no ``self``, no agent dependency, no I/O. Safe to call from
    offline tools, tests, and any future non-agent context.
    """
    if context_keys:
        return RewardPipeline([PerContextZScore(context_keys=context_keys)])
    return RewardPipeline([GlobalMeanStdFilter()])
