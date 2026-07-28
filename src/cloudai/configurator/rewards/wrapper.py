# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Library-agnostic env wrapper that injects a :class:`RewardPipeline`.

Sits between the environment and the RL library. On every ``step()`` the
wrapper:

1. Forwards the action to the wrapped env.
2. Calls ``context_provider()`` to fetch the current trial's raw context
   (e.g. ``{"drop_rate": 0.0}``) and projects it onto ``context_keys``.
3. Calls ``pipeline.step(reward, ctx)`` to obtain the transformed reward.
4. Stashes the original ``reward`` under ``info["raw_reward"]`` so reports
   and downstream tooling that need the un-normalized signal can recover
   it without re-running the workload.

Inherits from :class:`gymnasium.Wrapper` when gymnasium is available so
the wrapper passes RLlib's registry-side ``isinstance(env, gym.Env)``
check. Falls back to a plain-Python base when gymnasium is absent so the
class is still importable from environments that don't pull gymnasium
in.

Why a wrapper instead of an RLlib ``ConnectorV2``: the reward
transformation logic is RL-library-agnostic. Putting it in a connector
would mean reimplementing the same logic for every future RL backend
(CleanRL, SB3, custom torch loops). A ``gym.Env``-style wrapper is the
single place every backend already understands.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

from .pipeline import RewardPipeline

logger = logging.getLogger(__name__)

try:  # noqa: SIM105 — explicit branching is clearer than contextlib.suppress
    import gymnasium as _gym

    _BaseWrapper: Any = _gym.Wrapper
except ImportError:  # pragma: no cover — gymnasium is a soft dep of cloudai
    _BaseWrapper = object


class PipelineRewardWrapper(_BaseWrapper):
    """Replace per-step reward with ``pipeline.step(reward, ctx)``.

    Parameters
    ----------
    env :
        Any object with a gym/gymnasium-style ``step(action)`` returning a
        4- or 5-tuple. ``__init__`` does NOT call
        ``gymnasium.Wrapper.__init__`` because that path runs an
        ``isinstance(env, gym.Env)`` check that would reject duck-typed
        envs like :class:`GymnasiumAdapter`. We assign ``self.env``
        directly instead -- gymnasium's helpers (``unwrapped``,
        ``observation_space`` / ``action_space`` forwarding) only need
        the attribute, not a typed init.
    pipeline :
        A :class:`RewardPipeline` whose transforms have already been
        configured (typically via :func:`build_default_pipeline`).
    context_provider :
        Zero-arg callable returning the current trial's raw context as a
        dict. Called once per ``step()`` AFTER the env step has executed,
        so the dict reflects the env_params used to compute *this* step's
        reward (matches how :class:`EnvParamsObserver` populates
        ``current_env_params`` before ``compute_reward`` runs).
    context_keys :
        Optional projection. When non-empty, only these keys are forwarded
        to the pipeline. When empty, the full context dict is forwarded
        unchanged. The default :class:`PerContextZScore` transform expects
        every declared key to be present in ``ctx`` -- the projection
        guarantees that contract even when the env exposes additional
        env_params the reward pipeline does not key on.
    """

    def __init__(
        self,
        env: Any,
        pipeline: RewardPipeline,
        context_provider: Callable[[], Mapping[str, Any]],
        context_keys: tuple[str, ...] = (),
    ) -> None:
        # Set the gymnasium-canonical attr name. Any consumer that walks
        # ``self.env`` (gym.Wrapper.unwrapped, RLlib's env_runner, etc.)
        # finds the underlying env here.
        self.env = env
        self.pipeline = pipeline
        self._context_provider = context_provider
        self._context_keys = tuple(context_keys)
        # Mirror gymnasium.Wrapper attributes that some RLlib paths read
        # without going through __getattr__. Forwarding via spaces keeps
        # the wrapper observationally identical to a plain gym.Wrapper.
        self.observation_space = getattr(env, "observation_space", None)
        self.action_space = getattr(env, "action_space", None)
        self.metadata = getattr(env, "metadata", {"render_modes": []})
        # spec is a read-only property on gymnasium.Wrapper backed by
        # ``self.env.spec``; nothing to set on this side.

    @property
    def unwrapped(self) -> Any:
        return getattr(self.env, "unwrapped", self.env)

    def __getattr__(self, name: str) -> Any:
        # Delegate any unmapped attribute to the wrapped env. Triggered
        # only when normal attribute lookup misses on this instance, so
        # the explicit attributes set in __init__ take precedence.
        #
        # Special case: dunder names (``__reduce__``, ``__getstate__``,
        # ``__setstate__``, ``__copy__``, ...) MUST NOT be forwarded.
        # During pickling (Ray cloudpickle wraps the env in
        # ``register_env``), the unpickler probes these names BEFORE
        # ``self.env`` is restored, which would recurse infinitely.
        # Raising ``AttributeError`` lets the standard pickle protocol
        # fall back to the default ``__dict__``-based path.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        # During unpickling self.env may not yet be set; guard explicitly.
        env = self.__dict__.get("env")
        if env is None:
            raise AttributeError(name)
        return getattr(env, name)

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        return self.env.reset(*args, **kwargs)

    def step(self, action: Any) -> tuple:
        result = self.env.step(action)
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            transformed, raw = self._apply_pipeline(reward)
            info = self._inject_raw(info, raw)
            return obs, transformed, terminated, truncated, info
        if len(result) == 4:
            obs, reward, done, info = result
            transformed, raw = self._apply_pipeline(reward)
            info = self._inject_raw(info, raw)
            return obs, transformed, done, info
        raise ValueError(
            f"PipelineRewardWrapper expects env.step() to return a 4- or 5-tuple; "
            f"got tuple of length {len(result)}"
        )

    def render(self, *args: Any, **kwargs: Any) -> Any:
        return self.env.render(*args, **kwargs)

    def close(self) -> Any:
        return self.env.close() if hasattr(self.env, "close") else None

    def _apply_pipeline(self, reward: Any) -> tuple[float, float]:
        raw = float(reward)
        ctx = self._extract_context()
        transformed = float(self.pipeline.step(raw, ctx))
        self._log_step(raw, transformed, ctx)
        return transformed, raw

    def _log_step(self, raw: float, transformed: float, ctx: Mapping[str, Any]) -> None:
        """Emit one verifiable per-step line describing the transform.

        Logs at INFO so it lands in the standard run log without enabling
        DEBUG globally. Each transform that exposes a ``snapshot`` is asked
        for its current stats so an operator can confirm
        ``z = (raw − mean) / std`` by hand and see whether per-context
        normalization is ``active`` yet (vs. still in warm-up).

        Guarded by ``logger.isEnabledFor`` so the snapshot work is skipped
        entirely when the level is suppressed.
        """
        if not logger.isEnabledFor(logging.INFO):
            return
        parts: list[str] = []
        for t in self.pipeline.transforms:
            snap = getattr(t, "snapshot", None)
            if snap is None:
                continue
            name = type(t).__name__
            try:
                if name == "PerContextZScore":
                    s = snap(ctx)
                    parts.append(
                        f"{name}[bin={s.get('bin')} n={s.get('count')} "
                        f"mean={s.get('mean'):.4f} std={s.get('std'):.4f} "
                        f"active={s.get('active')}]"
                    )
                else:
                    s = snap()
                    parts.append(
                        f"{name}[n={s.get('count')} mean={s.get('mean'):.4f} "
                        f"std={s.get('std'):.4f} active={s.get('active')}]"
                    )
            except Exception:  # noqa: BLE001 — logging must never break a step
                parts.append(f"{name}[snapshot-error]")
        logger.info(
            "[reward-pipeline] ctx=%s raw=%.6f -> reward=%.6f delta=%.6f | %s",
            dict(ctx),
            raw,
            transformed,
            transformed - raw,
            " ".join(parts) if parts else "(no stateful transforms)",
        )

    def _extract_context(self) -> dict[str, Any]:
        raw_ctx = self._context_provider() or {}
        if not self._context_keys:
            return dict(raw_ctx)
        return {k: raw_ctx[k] for k in self._context_keys if k in raw_ctx}

    def _inject_raw(self, info: Any, raw_reward: float) -> dict[str, Any]:
        merged: dict[str, Any] = dict(info or {})
        merged["raw_reward"] = raw_reward
        return merged
