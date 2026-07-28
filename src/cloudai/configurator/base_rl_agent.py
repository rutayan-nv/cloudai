# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Abstract base class shared by all RL agent implementations.

Sits between :class:`BaseAgent` and concrete library adapters
(``RLlibPPOAgent``, ``CleanRLPPOAgent``, ``SB3PPOAgent``, ...). Owns the
reward pipeline AND the bridge that injects it into the env. The bridge
is intentionally library-agnostic: a single
:class:`PipelineRewardWrapper` that behaves like a gym/gymnasium env
wrapper. RLlib, CleanRL, SB3, and custom torch loops all consume gym
envs, so one bridge covers every backend.

Lifecycle (subclasses normally call ``super()`` from each step):

1. ``__init__``       -- store env/config (inherited from BaseAgent)
2. ``setup_rewards``  -- auto-detect context keys; build pipeline
3. ``wrap_env(env)``  -- wrap env with :class:`PipelineRewardWrapper`
4. ``train()``        -- subclass: lib-specific training loop

Subclasses do NOT need to override ``wrap_env`` for the standard reward
shaping path; the default returns the right wrapper. They may override
to chain additional library-specific wrappers AROUND the reward wrapper
(e.g., RLlib registers the wrapped env via ``register_env``).
"""

from __future__ import annotations

from typing import Any

from .base_agent import BaseAgent, BaseAgentConfig
from .base_gym import BaseGym
from .rewards import (
    ContextAutoDetector,
    PipelineRewardWrapper,
    RewardPipeline,
    build_default_pipeline,
)


class RLAgentBase(BaseAgent):
    """Common scaffolding for RL agents across libraries.

    Subclasses implement ``train`` (lib-specific loop) and call
    :meth:`setup_rewards` from ``__init__``. Everything else is shared:
    context auto-detection, pipeline construction, the default reward
    bridge via :class:`PipelineRewardWrapper`, and cold-starting the
    pipeline on every fresh process.
    """

    pipeline: RewardPipeline
    _context_keys: tuple[str, ...]

    def __init__(self, env: BaseGym, config: BaseAgentConfig) -> None:
        super().__init__(env, config)
        self.pipeline = RewardPipeline()
        self._context_keys = ()

    def setup_rewards(self) -> None:
        """Build the reward pipeline. Cold start; no warm-up from reports.

        Concrete subclasses call this exactly once, before training begins.
        Default policy (Design X — one estimator, selected from context
        structure, never chained):

        * ``ContextAutoDetector`` returns non-empty context keys (a
          low-cardinality categorical context) → a single ``PerContextZScore``
          on the raw reward.
        * Otherwise → a single ``GlobalMeanStdFilter`` (regime-blind).

        Side-effects: populates ``self.pipeline`` and ``self._context_keys``
        so :meth:`wrap_env` can build a correctly-keyed reward wrapper.
        """
        detector = ContextAutoDetector(self.env)
        context_keys = detector.initial_context_keys()
        self._context_keys = context_keys
        self.pipeline = self._build_pipeline(context_keys)

    def _build_pipeline(self, context_keys: tuple[str, ...]) -> RewardPipeline:
        """Default transform stack. Override to customize.

        Thin hook over :func:`build_default_pipeline` so subclasses that want
        a non-default stack can override this one method without re-stating
        the whole construction. The free function carries the actual logic.
        """
        return build_default_pipeline(context_keys)

    def wrap_env(self, env: Any) -> Any:
        """Wrap ``env`` so each step's reward is routed through the pipeline.

        Returns ``env`` unchanged when the pipeline carries no transforms
        (no ``setup_rewards()`` call yet, or the pipeline was explicitly
        cleared). Otherwise returns a :class:`PipelineRewardWrapper` that
        calls ``self.pipeline.step(reward, ctx)`` per step and stashes
        the original reward in ``info["raw_reward"]``.

        The wrapped env preserves the gymnasium env contract (5-tuple
        step, reset/render/close passthrough, observation/action space
        forwarding) so downstream RL libraries see no behavioral
        difference beyond the rewritten reward.

        Subclasses may override to chain additional library-specific
        wrappers AROUND the reward wrapper -- e.g. RLlib's
        ``GymnasiumAdapter`` and ``_ContextualDSEEnv`` go BENEATH the
        reward wrapper so the pipeline sees the same per-step reward the
        learner sees.
        """
        if not self.pipeline.transforms:
            return env
        return PipelineRewardWrapper(
            env,
            self.pipeline,
            context_provider=self._context_provider(),
            context_keys=self._context_keys,
        )

    def _context_provider(self):
        """Resolve the callable that returns the current trial's context.

        Default: walks down to the innermost env via the gym ``unwrapped``
        chain (if any) and returns its :meth:`BaseGym.current_context`
        bound method. Subclasses can override to point at a different
        source (e.g., a non-BaseGym env that exposes context elsewhere).
        """
        env = self.env
        unwrapped = getattr(env, "unwrapped", env)
        target = unwrapped if isinstance(unwrapped, BaseGym) else env
        return target.current_context
