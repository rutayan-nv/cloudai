# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Domain-randomization primitives for CloudAI DSE.

An env-randomized parameter is a workload knob whose value the environment
samples per trial (categorical, optional weights). It is sibling to
``cmd_args`` on a ``TestDefinition`` and does not enter the agent's action
space; the policy learns a robust mapping under that variation.

This module owns the data schema (``EnvParamSpec``), the deterministic
sampler (``EnvParamsSampler``), the persistence interface
(``EnvParamsSink`` + ``CsvSink``) and the per-step observer
(``EnvParamsObserver``). ``CloudAIGymEnv`` consumes these directly so the
artifacts (``env.csv``) and the cache key align 1:1 with ``trajectory.csv``
regardless of agent (PPO, BO, GA, MAB) or workload.
"""

from __future__ import annotations

import csv
import dataclasses
import math
import random
from pathlib import Path
from typing import Annotated, Any, Dict, List, Literal, Optional, Protocol, Union, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

# ---------------------------------------------------------------------------
# Sampling specs (how the environment DRAWS a per-trial value)
# ---------------------------------------------------------------------------


class CategoricalSampling(BaseModel):
    """Draw from a finite candidate set, optionally weighted."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["categorical"] = "categorical"
    values: List[Any] = Field(
        min_length=2,
        description="Candidate values; for a fixed (non-randomized) parameter use a bare scalar instead.",
    )
    weights: Optional[List[float]] = Field(
        default=None,
        description="Optional probability weights aligned with values; uniform if omitted.",
    )

    @model_validator(mode="after")
    def _validate_weights(self) -> Self:
        if self.weights is None:
            return self
        if len(self.weights) != len(self.values):
            raise ValueError(
                f"env_params weights length {len(self.weights)} does not match values length {len(self.values)}"
            )
        for w in self.weights:
            if w < 0:
                raise ValueError(f"env_params weights must be non-negative; got {w}")
        if sum(self.weights) <= 0:
            raise ValueError("env_params weights must have a positive sum")
        return self


class LogUniformSampling(BaseModel):
    """
    Draw uniformly in log10 space over ``[low, high]`` with an optional zero mixture.

    ``zero_prob`` reserves probability mass for an exact ``0.0`` draw (e.g. a
    "no drop" baseline) before the continuous log-uniform body is sampled.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["loguniform"]
    low: float = Field(gt=0.0, description="Lower bound of the continuous body (strictly positive; log domain).")
    high: float = Field(gt=0.0, description="Upper bound of the continuous body.")
    zero_prob: float = Field(default=0.0, ge=0.0, le=1.0, description="Probability mass placed on an exact 0.0 draw.")

    @model_validator(mode="after")
    def _validate_range(self) -> Self:
        if self.high <= self.low:
            raise ValueError(f"loguniform high ({self.high}) must be greater than low ({self.low})")
        return self


class UniformSampling(BaseModel):
    """Draw uniformly in linear space over ``[low, high]``."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["uniform"]
    low: float
    high: float

    @model_validator(mode="after")
    def _validate_range(self) -> Self:
        if self.high <= self.low:
            raise ValueError(f"uniform high ({self.high}) must be greater than low ({self.low})")
        return self


class FixedSampling(BaseModel):
    """A fixed (non-randomized) value: the sampler always returns ``value``.

    Internal representation produced when a user writes a bare scalar for an
    env_param (e.g. ``drop_rate = 0.0``). Users do NOT need to spell this
    type out; ``EnvParamSpec`` accepts a scalar at parse time and lifts it
    here. The env_params machinery (sampler / sink / observer / observation
    pipeline) runs identically -- the only thing this changes is that the
    sampler returns the configured value unchanged each trial, without
    consulting any RNG.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["fixed"] = "fixed"
    value: Any = Field(description="The fixed value returned for every trial.")


Sampling = Annotated[
    Union[CategoricalSampling, LogUniformSampling, UniformSampling, FixedSampling],
    Field(discriminator="type"),
]


_SCALAR_TYPES = (int, float, str)


# ---------------------------------------------------------------------------
# Encoding specs (how a raw value is PRESENTED to the policy as an obs leaf)
# ---------------------------------------------------------------------------


class LinearEncoding(BaseModel):
    """Pass the raw value through unchanged -> ``Box(1)``."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["linear"] = "linear"


class LogEncoding(BaseModel):
    """
    Encode an exponential-scale value as ``[is_zero, log10(max(x, floor))]`` -> ``Box(2)``.

    The ``is_zero`` indicator separates an exact-zero baseline from the
    continuous body; ``floor`` (default: the sampler's ``low``) anchors the
    log term so a zero draw lands at the bottom of the range instead of at an
    extreme outlier.
    """

    model_config = ConfigDict(extra="forbid")
    type: Literal["log"] = "log"
    floor: Optional[float] = Field(default=None, gt=0.0, description="Log floor; defaults to sampling.low.")


class AsinhEncoding(BaseModel):
    """Signed log-like encoding ``asinh(x / scale)`` -> ``Box(1)``; handles zero/negatives."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["asinh"] = "asinh"
    scale: float = Field(default=1e-3, gt=0.0)


class CategoricalEncoding(BaseModel):
    """Present a categorical value as a ``Discrete(k)`` index (one-hot after flattening)."""

    model_config = ConfigDict(extra="forbid")
    type: Literal["categorical"] = "categorical"


Encoding = Annotated[
    Union[LinearEncoding, LogEncoding, AsinhEncoding, CategoricalEncoding],
    Field(discriminator="type"),
]


@dataclasses.dataclass(frozen=True)
class ObsLeafDescriptor:
    """
    Framework-agnostic description of one observation leaf.

    The gymnasium-aware adapter turns this into a concrete subspace (``Box``
    for ``kind="box"`` of width ``dim``; ``Discrete(n)`` for
    ``kind="discrete"``), keeping ``cloudai`` core free of a hard gymnasium
    dependency.
    """

    kind: Literal["box", "discrete"]
    dim: int = 1
    n: Optional[int] = None


class EnvParamSpec(BaseModel):
    """
    Specification of one env-randomized parameter: how it is sampled and observed.

    Two encapsulated, discriminated sub-blocks:

    * ``sampling`` — the distribution the environment draws from per trial.
    * ``encoding`` — how the drawn raw value is presented to the policy as an
      observation leaf. Optional; inferred from ``sampling`` when omitted.

    Backward compatibility: a bare ``values = [...]`` (+ optional ``weights``)
    is accepted as shorthand for ``sampling = {type = "categorical", ...}``
    with an inferred encoding, preserving the legacy schema unchanged.
    """

    model_config = ConfigDict(extra="forbid")

    sampling: Sampling
    encoding: Optional[Encoding] = Field(
        default=None,
        description="Observation encoding; inferred from sampling when omitted.",
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_scalar_or_legacy(cls, data: Any) -> Any:
        """Accept a bare scalar (fixed value) or lift legacy flat ``values``/``weights``.

        The public interface decouples ``fixed value`` from ``distribution``:
        a user writes ``drop_rate = 0.0`` for a constant (no randomization)
        and ``drop_rate = { sampling = { ... }, encoding = { ... } }`` for a
        randomized parameter. The sampler / sink / observer code path is
        unchanged; ``FixedSampling`` just short-circuits the draw step.
        """
        if isinstance(data, bool):
            return {"sampling": {"type": "fixed", "value": data}}
        if isinstance(data, _SCALAR_TYPES):
            return {"sampling": {"type": "fixed", "value": data}}
        if not isinstance(data, dict):
            return data
        if "values" in data or "weights" in data:
            if "sampling" in data:
                raise ValueError("env_params: provide either 'sampling' or flat 'values'/'weights', not both")
            sampling: Dict[str, Any] = {"type": "categorical"}
            if "values" in data:
                sampling["values"] = data["values"]
            if "weights" in data:
                sampling["weights"] = data["weights"]
            data = {k: v for k, v in data.items() if k not in ("values", "weights")}
            data["sampling"] = sampling
        return data

    @model_validator(mode="after")
    def _infer_and_validate_encoding(self) -> Self:
        if self.encoding is None:
            self.encoding = self._default_encoding()
        if isinstance(self.encoding, CategoricalEncoding) and not isinstance(self.sampling, CategoricalSampling):
            raise ValueError("categorical encoding requires categorical sampling (a finite value set)")
        return self

    def _default_encoding(self) -> Encoding:
        """
        Infer an encoding from the sampling type, preserving today's behaviour.

        categorical + numeric values -> linear (raw passthrough, unchanged);
        categorical + non-numeric -> categorical; fixed scalar -> linear;
        continuous -> linear.
        """
        if isinstance(self.sampling, CategoricalSampling):
            if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in self.sampling.values):
                return LinearEncoding()
            return CategoricalEncoding()
        return LinearEncoding()

    # -- legacy shorthand accessors (read-only) --------------------------------

    @property
    def values(self) -> Optional[List[Any]]:
        """Candidate values for finite samplers; ``None`` for continuous samplers.

        - ``CategoricalSampling``: the candidate list.
        - ``FixedSampling``: a singleton ``[value]`` (one-element list view of
          the constant), so legacy consumers that iterate over ``spec.values``
          treat a fixed value as a one-element finite set.
        - Otherwise: ``None``.
        """
        if isinstance(self.sampling, CategoricalSampling):
            return self.sampling.values
        if isinstance(self.sampling, FixedSampling):
            return [self.sampling.value]
        return None

    @property
    def weights(self) -> Optional[List[float]]:
        """Categorical weights, if any."""
        return self.sampling.weights if isinstance(self.sampling, CategoricalSampling) else None

    # -- observation encoding (env-owned) -------------------------------------

    def observation_descriptor(self) -> ObsLeafDescriptor:
        """Describe the observation leaf this parameter contributes."""
        enc = self.encoding
        if isinstance(enc, LogEncoding):
            return ObsLeafDescriptor(kind="box", dim=2)
        if isinstance(enc, CategoricalEncoding):
            assert isinstance(self.sampling, CategoricalSampling)
            return ObsLeafDescriptor(kind="discrete", dim=1, n=len(self.sampling.values))
        return ObsLeafDescriptor(kind="box", dim=1)

    def encode_observation(self, raw: Any) -> Any:
        """
        Encode a drawn raw value into its observation leaf.

        Returns a ``list[float]`` for ``box`` leaves and an ``int`` index for
        ``categorical`` (``discrete``) leaves.
        """
        enc = self.encoding
        if isinstance(enc, LinearEncoding):
            return [float(raw)]
        if isinstance(enc, AsinhEncoding):
            return [math.asinh(float(raw) / enc.scale)]
        if isinstance(enc, LogEncoding):
            x = float(raw)
            floor = enc.floor if enc.floor is not None else self._log_floor()
            is_zero = 1.0 if x <= 0.0 else 0.0
            return [is_zero, math.log10(max(x, floor))]
        if isinstance(enc, CategoricalEncoding):
            assert isinstance(self.sampling, CategoricalSampling)
            return self.sampling.values.index(raw)
        raise TypeError(f"Unsupported encoding: {enc!r}")

    def _log_floor(self) -> float:
        """Pick a positive log floor from the sampling spec."""
        if isinstance(self.sampling, LogUniformSampling):
            return self.sampling.low
        if isinstance(self.sampling, CategoricalSampling):
            positives = [float(v) for v in self.sampling.values if isinstance(v, (int, float)) and float(v) > 0.0]
            if positives:
                return min(positives)
        if isinstance(self.sampling, FixedSampling):
            v = self.sampling.value
            if isinstance(v, (int, float)) and not isinstance(v, bool) and float(v) > 0.0:
                return float(v)
        return 1e-6


class EnvParamsSampler:
    """
    Per-trial sampler dispatching on each parameter's sampling distribution.

    Determinism contract: ``sample(t)`` returns the same dict on every call
    (across processes) for the same ``(seed, env_params, t)``.

    Independence contract: each parameter uses an RNG seeded by
    ``f"{seed}:{name}:{trial}"`` so adding or removing an unrelated
    parameter does not perturb existing parameters' draw sequences.
    """

    def __init__(self, env_params: Dict[str, EnvParamSpec], seed: int) -> None:
        self._env_params = env_params
        self._seed = seed

    def sample(self, trial: int) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name, spec in self._env_params.items():
            rng = random.Random(f"{self._seed}:{name}:{trial}")
            out[name] = self._draw(spec.sampling, rng)
        return out

    @staticmethod
    def _draw(sampling: Any, rng: random.Random) -> Any:
        if isinstance(sampling, FixedSampling):
            return sampling.value
        if isinstance(sampling, CategoricalSampling):
            if sampling.weights is not None:
                return rng.choices(sampling.values, weights=sampling.weights, k=1)[0]
            return rng.choice(sampling.values)
        if isinstance(sampling, LogUniformSampling):
            if sampling.zero_prob > 0.0 and rng.random() < sampling.zero_prob:
                return 0.0
            return 10.0 ** rng.uniform(math.log10(sampling.low), math.log10(sampling.high))
        if isinstance(sampling, UniformSampling):
            return rng.uniform(sampling.low, sampling.high)
        raise TypeError(f"Unsupported sampling: {sampling!r}")


@runtime_checkable
class EnvParamsSink(Protocol):
    """Persist one trial's env_params sample; empty samples must be no-ops."""

    def write(self, step: int, sample: Dict[str, Any]) -> None: ...


class CsvSink:
    """
    Append per-trial env_params samples to a step-aligned CSV.

    The CSV mirrors how ``trajectory.csv`` serialises its ``action`` column
    (one row per env.step(), sample dict stringified in a single cell) so the
    two files align 1:1 on ``step`` and a plain ``merge`` joins them.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    def write(self, step: int, sample: Dict[str, Any]) -> None:
        if step < 1:
            raise ValueError(f"step must be a positive trial index (cloudai DSE is 1-based); got {step}")
        if not sample:
            return
        new_file = not self._path.exists()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", newline="") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(("step", "env"))
            writer.writerow([step, sample])


@runtime_checkable
class StepObserver(Protocol):
    """
    Hook fired by ``CloudAIGymEnv.step()`` around each trial.

    ``before_step`` runs before the cache lookup and before any workload
    execution. ``after_step`` runs after the trajectory row is written.
    """

    def before_step(self, test_run: Any) -> None: ...

    def after_step(self, test_run: Any, observation: list, reward: float) -> None: ...


class EnvParamsObserver:
    """
    StepObserver that samples env_params per step and persists them.

    Pre-step: samples ``test_run.test.env_params`` for ``test_run.step``,
    stashes the result on ``test_run.current_env_params`` (so the cache key
    and the workload's substitution both see it), and appends a row to
    ``env.csv``. Post-step: no-op (trajectory.csv is written by CloudAIGymEnv).
    """

    def __init__(self, env_params: Dict[str, EnvParamSpec], sink: EnvParamsSink, seed: int) -> None:
        self._sampler = EnvParamsSampler(env_params, seed=seed)
        self._sink = sink

    def before_step(self, test_run: Any) -> None:
        sample = self._sampler.sample(test_run.step)
        test_run.current_env_params = sample
        self._sink.write(test_run.step, sample)

    def after_step(self, test_run: Any, observation: list, reward: float) -> None:
        del test_run, observation, reward  # no-op; trajectory.csv handled by env
