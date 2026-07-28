# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Reward transforms: composable units used by :class:`RewardPipeline`.

Contract
--------
Each transform implements::

    update(reward, ctx)  -> None    # ingest one observation; mutate state
    apply(reward, ctx)   -> float   # produce transformed reward; do not mutate

Pipeline call order per env step is ``update`` then ``apply`` (see
:class:`RewardPipeline`). Transforms must be safe under "update before apply"
on the **same** sample.

State model
-----------
* **Stateless config** is set at ``__init__`` and never changes.
* **Stateful state** (running stats, counters) lives in private attributes,
  is held in memory only, and is **not** persisted to the env's reporting
  plane.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable


def _clamp(value: float, clip: float | None) -> float:
    """Clamp ``value`` to ``[−clip, +clip]``; no-op when ``clip`` is None.

    Used by the normalizing transforms to cap a pathological warm-up z-score
    (tiny early std → |z| ≈ 10) so a single bad sample cannot dominate the
    policy gradient. Inert in steady state where real z stays within a few σ.
    """
    if clip is None:
        return value
    return max(-clip, min(clip, value))


@runtime_checkable
class RewardTransform(Protocol):
    """Protocol for a single reward transform.

    Pipelines compose multiple transforms in order; see :class:`RewardPipeline`.
    """

    def update(self, reward: float, ctx: Mapping[str, Any]) -> None:
        """Ingest one (reward, context) tuple to update internal state."""
        ...

    def apply(self, reward: float, ctx: Mapping[str, Any]) -> float:
        """Return the transformed reward without further mutating state."""
        ...


class IdentityTransform:
    """Pass-through. Default for non-RL agents and for RL when no
    normalization is required (e.g., gating heuristic returns False)."""

    def update(self, reward: float, ctx: Mapping[str, Any]) -> None:
        return None

    def apply(self, reward: float, ctx: Mapping[str, Any]) -> float:
        return reward


class GlobalMeanStdFilter:
    """Single global running mean / variance with EMA momentum.

    Counterpart to RLlib's ``MeanStdFilter`` for rewards. Useful when
    rewards have non-zero mean or non-unit variance but no per-context
    structure dominates (η² below the auto-promote threshold).
    """

    def __init__(
        self,
        momentum: float = 0.99,
        epsilon: float = 1e-6,
        clip: float | None = 10.0,
    ) -> None:
        self._momentum = float(momentum)
        self._epsilon = float(epsilon)
        self._clip = None if clip is None else float(clip)
        self._mean: float = 0.0
        self._var: float = 1.0
        self._count: int = 0

    def update(self, reward: float, ctx: Mapping[str, Any]) -> None:
        """EMA update of global (mean, var, count). ``ctx`` is ignored.

        Init (first sample) sets ``mean = r, var = 0`` so the EMA is
        unbiased — same pattern as :class:`PerContextZScore`. Subsequent
        samples apply the same recursion::

            mean ← α·mean + (1−α)·r
            var  ← α·var  + (1−α)·(r − mean_old)²
        """
        if self._count == 0:
            self._mean = float(reward)
            self._var = 0.0
            self._count = 1
            return
        alpha = self._momentum
        mean_old = self._mean
        self._mean = alpha * mean_old + (1.0 - alpha) * reward
        self._var = alpha * self._var + (1.0 - alpha) * (reward - mean_old) ** 2
        self._count += 1

    def apply(self, reward: float, ctx: Mapping[str, Any]) -> float:
        """Global z-score, with warm-up gating.

        * ``count == 0`` → no samples yet → return ``reward`` unchanged.
        * ``count == 1`` → ``var == 0`` → dividing by ``√ε`` would explode;
          pass through. Normalization is meaningful from the second sample
          onward.
        * Otherwise → ``(reward − mean) / √(var + ε)``, clamped to
          ``[−clip, +clip]`` when ``clip`` is set (bounds warm-up spikes).
        """
        if self._count < 2:
            return float(reward)
        z = (float(reward) - self._mean) / math.sqrt(self._var + self._epsilon)
        return _clamp(z, self._clip)

    def snapshot(self) -> dict[str, Any]:
        """Observability hook: current running stats (read-only).

        Returns ``{"count", "mean", "std", "active"}`` where ``active`` is
        ``True`` once ``apply`` starts normalizing (``count >= 2``). Used by
        :class:`PipelineRewardWrapper` to emit a verifiable per-step log so
        operators can confirm ``z = (raw − mean) / std`` by hand.
        """
        return {
            "count": self._count,
            "mean": self._mean,
            "std": math.sqrt(self._var + self._epsilon),
            "active": self._count >= 2,
        }


class PerContextZScore:
    """Per-context running z-score normalization.

    Bins by the tuple of values addressed by ``context_keys``. Each bin
    maintains its own (mean, var, count) under EMA momentum. While a bin
    has fewer than ``min_samples`` updates, ``apply`` falls through to the
    identity to avoid amplifying noise during warm-up.

    Stateless config: ``context_keys``, ``momentum``, ``epsilon``, ``min_samples``.
    Stateful state:   per-bin ``(mean, var, count)`` in ``self._stats``.

    See ``cloudai.configurator.rewards`` package docstring for rationale and
    auto-promotion criteria (ANOVA significance + advantage sign-flip check).
    """

    def __init__(
        self,
        context_keys: tuple[str, ...],
        momentum: float = 0.99,
        epsilon: float = 1e-6,
        min_samples: int = 5,
        clip: float | None = 10.0,
    ) -> None:
        self._keys: tuple[str, ...] = tuple(context_keys)
        self._momentum: float = float(momentum)
        self._epsilon: float = float(epsilon)
        self._min_samples: int = int(min_samples)
        self._clip = None if clip is None else float(clip)
        self._stats: dict[tuple[Any, ...], tuple[float, float, int]] = {}

    def _bin_key(self, ctx: Mapping[str, Any]) -> tuple[Any, ...]:
        """Compose the per-bin key by reading ``self._keys`` from ``ctx``.

        Order is fixed by ``context_keys`` so that the bin mapping is stable
        regardless of dict insertion order in ``ctx``. Raises ``KeyError``
        when a configured key is missing — that's a programming error in the
        env (it failed to expose a context value the agent expected).
        """
        return tuple(ctx[k] for k in self._keys)

    def update(self, reward: float, ctx: Mapping[str, Any]) -> None:
        """EMA update of per-bin (mean, var, count).

        Init (first sample for a bin)::

            (mean, var, count) ← (r, 0, 1)

        Subsequent samples::

            mean ← α·mean + (1−α)·r          (α = momentum)
            var  ← α·var  + (1−α)·(r − mean_old)²
            count ← count + 1

        Initialising ``mean`` from the first sample (rather than 0) makes
        the EMA estimator unbiased: ``E[mean_N] = μ`` for i.i.d. samples
        from ``N(μ, σ²)``, even at small N.
        """
        key = self._bin_key(ctx)
        prior = self._stats.get(key)
        if prior is None:
            self._stats[key] = (float(reward), 0.0, 1)
            return
        mean_old, var_old, count = prior
        alpha = self._momentum
        mean_new = alpha * mean_old + (1.0 - alpha) * reward
        var_new = alpha * var_old + (1.0 - alpha) * (reward - mean_old) ** 2
        self._stats[key] = (mean_new, var_new, count + 1)

    def apply(self, reward: float, ctx: Mapping[str, Any]) -> float:
        """Per-bin z-score, with warm-up gating.

        * Bin not seen yet → return ``reward`` unchanged. Avoids dividing by
          zero / amplifying noise on a brand-new regime.
        * ``count < min_samples`` → return ``reward`` unchanged. Same
          rationale: the running stats haven't converged enough to trust.
        * Otherwise → ``(reward − mean) / √(var + ε)``, clamped to
          ``[−clip, +clip]`` when ``clip`` is set (bounds warm-up spikes).
        """
        key = self._bin_key(ctx)
        prior = self._stats.get(key)
        if prior is None:
            return float(reward)
        mean, var, count = prior
        if count < self._min_samples:
            return float(reward)
        z = (float(reward) - mean) / math.sqrt(var + self._epsilon)
        return _clamp(z, self._clip)

    def snapshot(self, ctx: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Observability hook: per-bin running stats (read-only).

        With ``ctx``: returns the single bin addressed by ``ctx`` as
        ``{"bin", "count", "mean", "std", "active"}`` where ``active`` means
        ``count >= min_samples`` (per-context z-score engaged). With no
        ``ctx``: returns ``{"bins": {bin_key: {...}}}`` for every bin seen.

        ``active=False`` is the verifiable signal that a regime is still in
        warm-up and the policy is therefore seeing the *upstream* reward for
        that regime, not yet a per-context z-score.
        """
        def _fmt(stat: tuple[float, float, int]) -> dict[str, Any]:
            mean, var, count = stat
            return {
                "count": count,
                "mean": mean,
                "std": math.sqrt(var + self._epsilon),
                "active": count >= self._min_samples,
            }

        if ctx is not None:
            key = self._bin_key(ctx)
            prior = self._stats.get(key)
            if prior is None:
                return {"bin": key, "count": 0, "mean": 0.0, "std": 0.0, "active": False}
            return {"bin": key, **_fmt(prior)}
        return {"bins": {k: _fmt(v) for k, v in self._stats.items()}}
