# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Context auto-detection for the reward pipeline.

Decides which observation entries qualify as *context* — values the agent
cannot influence (sampled from ``env_params``) that condition the optimal
action. Used by :class:`RLAgentBase` to construct a per-context normalization
transform when (and only when) the data justifies it.

Two-stage decision (data-light → data-driven):

* **Schema gate** (at agent init): pick low-cardinality categorical keys
  from ``env.test_run.test.env_params``. Fast; no rewards required.
* **Significance gate** (after a warm-up window): one-way ANOVA on the
  rewards observed so far, grouped by candidate context keys. Promote only
  if the F-test is significant (e.g., p < 0.01) and the implied advantage
  bias would flip sign for at least one regime under raw-reward training.

The schema gate alone is enough for a v1; the significance gate prevents
spurious activation on workloads where env_params don't materially affect
reward magnitude.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from cloudai.configurator.env_params import CategoricalSampling, EnvParamSpec


class ContextAutoDetector:
    """Inspects an env and decides which obs keys are 'context'.

    Designed to be cheap at init (schema-only) and idempotent at promotion
    time (schema + statistics). Returns ``()`` when no context is warranted,
    in which case the pipeline falls back to identity / global filter only.
    """

    def __init__(self, env: Any) -> None:
        self._env = env

    def initial_context_keys(
        self,
        max_cardinality: int = 16,
    ) -> tuple[str, ...]:
        """Schema-only candidates from ``env.test_run.test.env_params``.

        Returns the names of every env_param whose ``sampling`` is
        :class:`CategoricalSampling` with cardinality
        ``<= max_cardinality``. The order matches the dict insertion order
        of ``env_params`` (Python guarantees this since 3.7).

        Continuous samplings (``LogUniformSampling`` / ``UniformSampling``)
        are excluded — binning a continuous distribution requires an extra
        discretization choice that doesn't belong in a schema gate.
        ``FixedSampling`` is excluded too — its single-value cardinality
        would collapse the per-context filter into a global one anyway.

        Returns an empty tuple when the env exposes no env_params or none
        of them qualify, in which case :func:`build_default_pipeline`
        falls back to identity normalization.
        """
        env_params = self._extract_env_params()
        out: list[str] = []
        for name, spec in env_params.items():
            sampling = getattr(spec, "sampling", None)
            if not isinstance(sampling, CategoricalSampling):
                continue
            if len(sampling.values) > max_cardinality:
                continue
            out.append(name)
        return tuple(out)

    def _extract_env_params(self) -> dict[str, EnvParamSpec]:
        """Walk the env to ``test_run.test.env_params`` defensively.

        Returns ``{}`` if any link in the chain is missing — the detector
        treats absence as 'no context candidates' rather than raising,
        because it must cope with envs that don't randomize anything.
        """
        test_run = getattr(self._env, "test_run", None)
        if test_run is None:
            return {}
        test = getattr(test_run, "test", None)
        if test is None:
            return {}
        env_params = getattr(test, "env_params", None)
        if not env_params:
            return {}
        return env_params

    def maybe_promote(
        self,
        candidate_keys: Sequence[str],
        rewards: Sequence[float],
        contexts: Sequence[dict[str, Any]],
        *,
        alpha: float = 0.01,
    ) -> tuple[str, ...]:
        """Data-driven gate after the warm-up window.

        Runs a one-way ANOVA on ``rewards`` grouped by the tuple of values
        from each candidate key. Returns the subset of keys that pass:

        1. ANOVA p-value below ``alpha``.
        2. At least one regime's mean differs from the global mean by a
           magnitude that would flip the sign of the implied advantage
           under raw-reward training (i.e., ``|mu_d - r_bar| > sigma_d/2``
           for some regime ``d``).

        Returns ``()`` when neither condition holds.
        """
        raise NotImplementedError(
            "skeleton: scipy.stats.f_oneway on grouped rewards; "
            "compute per-regime mu_d, sigma_d, r_bar; "
            "return keys passing both gates"
        )
