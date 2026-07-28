# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for the domain-randomization primitives in cloudai.configurator.env_params."""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
from pydantic import ValidationError

from cloudai.configurator.env_params import (
    AsinhEncoding,
    CategoricalEncoding,
    CategoricalSampling,
    CsvSink,
    EnvParamsObserver,
    EnvParamSpec,
    EnvParamsSampler,
    LinearEncoding,
    LogEncoding,
    LogUniformSampling,
)


class _RecordingSink:
    """Test double capturing every (step, sample) pair sent to the sink."""

    def __init__(self) -> None:
        self.calls: List[tuple[int, Dict[str, Any]]] = []

    def write(self, step: int, sample: Dict[str, Any]) -> None:
        self.calls.append((step, dict(sample)))


def test_env_param_spec_requires_at_least_two_values() -> None:
    with pytest.raises(ValidationError):
        EnvParamSpec(values=[0.0])


def test_env_param_spec_rejects_mismatched_weights() -> None:
    with pytest.raises(ValidationError):
        EnvParamSpec(values=[0.0, 0.1], weights=[1.0])


def test_env_param_spec_rejects_zero_sum_weights() -> None:
    with pytest.raises(ValidationError):
        EnvParamSpec(values=[0.0, 0.1], weights=[0.0, 0.0])


def test_sampler_is_deterministic_across_calls() -> None:
    spec = {"drop_rate": EnvParamSpec(values=[0.0, 0.001, 0.01])}
    a = EnvParamsSampler(spec, seed=42)
    b = EnvParamsSampler(spec, seed=42)
    seq_a = [a.sample(t) for t in range(1, 6)]
    seq_b = [b.sample(t) for t in range(1, 6)]
    assert seq_a == seq_b, "same (seed, trial) must produce the same draw across instances"


def test_sampler_each_param_is_independent() -> None:
    """Adding an unrelated parameter must not perturb existing parameters' draws."""
    base = {"drop_rate": EnvParamSpec(values=[0.0, 0.001, 0.01])}
    extended = {
        "drop_rate": EnvParamSpec(values=[0.0, 0.001, 0.01]),
        "latency_ms": EnvParamSpec(values=[1, 5, 10]),
    }
    a = [EnvParamsSampler(base, seed=7).sample(t)["drop_rate"] for t in range(1, 11)]
    b = [EnvParamsSampler(extended, seed=7).sample(t)["drop_rate"] for t in range(1, 11)]
    assert a == b, "per-parameter RNG seeding must isolate parameters from each other"


def test_csv_sink_skips_empty_samples_and_rejects_zero_step(tmp_path: Path) -> None:
    sink = CsvSink(tmp_path / "env.csv")
    sink.write(1, {})  # empty -> no-op, no file
    assert not (tmp_path / "env.csv").exists()
    with pytest.raises(ValueError):
        sink.write(0, {"drop_rate": 0.0})


def test_csv_sink_writes_header_then_rows(tmp_path: Path) -> None:
    sink = CsvSink(tmp_path / "env.csv")
    sink.write(1, {"drop_rate": 0.001})
    sink.write(2, {"drop_rate": 0.01})
    contents = (tmp_path / "env.csv").read_text().strip().splitlines()
    assert contents[0] == "step,env"
    assert contents[1].startswith("1,")
    assert contents[2].startswith("2,")


def test_observer_sets_current_env_params_and_persists_sample() -> None:
    spec = {"drop_rate": EnvParamSpec(values=[0.0, 0.001, 0.01])}
    sink = _RecordingSink()
    observer = EnvParamsObserver(spec, sink, seed=42)
    test_run = SimpleNamespace(step=3, current_env_params={})

    observer.before_step(test_run)

    assert "drop_rate" in test_run.current_env_params
    assert test_run.current_env_params["drop_rate"] in {0.0, 0.001, 0.01}
    assert sink.calls == [(3, dict(test_run.current_env_params))]


def test_observer_after_step_is_noop() -> None:
    """after_step must not touch test_run or sink; trajectory.csv handles persistence."""
    sink = _RecordingSink()
    observer = EnvParamsObserver({}, sink, seed=0)
    test_run = SimpleNamespace(step=1, current_env_params={"x": 1})

    observer.after_step(test_run, observation=[0.0], reward=0.0)

    assert sink.calls == []
    assert test_run.current_env_params == {"x": 1}


# --- legacy back-compat: flat values shorthand -> categorical sampling --------


def test_legacy_values_lifts_to_categorical_sampling() -> None:
    spec = EnvParamSpec(values=[0.0, 0.001, 0.01])
    assert isinstance(spec.sampling, CategoricalSampling)
    assert spec.sampling.values == [0.0, 0.001, 0.01]
    assert spec.values == [0.0, 0.001, 0.01]  # read-only shorthand accessor preserved


def test_legacy_numeric_values_infer_linear_encoding() -> None:
    """Numeric categorical preserves today's raw passthrough behaviour (linear)."""
    spec = EnvParamSpec(values=[0.0, 0.001, 0.01])
    assert isinstance(spec.encoding, LinearEncoding)


def test_non_numeric_values_infer_categorical_encoding() -> None:
    spec = EnvParamSpec(values=["bitmap_ooo", "count_threshold"])
    assert isinstance(spec.encoding, CategoricalEncoding)


def test_cannot_mix_sampling_and_legacy_values() -> None:
    with pytest.raises(ValidationError):
        EnvParamSpec(values=[0.0, 0.1], sampling={"type": "categorical", "values": [0.0, 0.1]})


# --- discriminated sampling/encoding unions -----------------------------------


def test_loguniform_spec_parses_and_validates_range() -> None:
    spec = EnvParamSpec(
        sampling={"type": "loguniform", "low": 1e-4, "high": 1e-1, "zero_prob": 0.2},
        encoding={"type": "log"},
    )
    assert isinstance(spec.sampling, LogUniformSampling)
    assert isinstance(spec.encoding, LogEncoding)
    assert spec.sampling.zero_prob == 0.2


def test_loguniform_rejects_inverted_range() -> None:
    with pytest.raises(ValidationError):
        EnvParamSpec(sampling={"type": "loguniform", "low": 1e-1, "high": 1e-4})


def test_loguniform_defaults_to_linear_encoding_when_omitted() -> None:
    spec = EnvParamSpec(sampling={"type": "loguniform", "low": 1e-4, "high": 1e-1})
    assert isinstance(spec.encoding, LinearEncoding)


def test_categorical_encoding_requires_categorical_sampling() -> None:
    with pytest.raises(ValidationError):
        EnvParamSpec(
            sampling={"type": "loguniform", "low": 1e-4, "high": 1e-1},
            encoding={"type": "categorical"},
        )


# --- sampler dispatch: loguniform + zero_prob mixture -------------------------


def test_loguniform_sampler_is_deterministic() -> None:
    spec = {"drop_rate": EnvParamSpec(sampling={"type": "loguniform", "low": 1e-4, "high": 1e-1, "zero_prob": 0.2})}
    a = [EnvParamsSampler(spec, seed=1).sample(t)["drop_rate"] for t in range(1, 20)]
    b = [EnvParamsSampler(spec, seed=1).sample(t)["drop_rate"] for t in range(1, 20)]
    assert a == b


def test_loguniform_sampler_respects_bounds_and_zero_mixture() -> None:
    spec = {"drop_rate": EnvParamSpec(sampling={"type": "loguniform", "low": 1e-4, "high": 1e-1, "zero_prob": 0.3})}
    sampler = EnvParamsSampler(spec, seed=7)
    draws = [sampler.sample(t)["drop_rate"] for t in range(1, 2001)]
    zeros = [d for d in draws if d == 0.0]
    nonzeros = [d for d in draws if d != 0.0]
    assert 0.2 < len(zeros) / len(draws) < 0.4, "zero_prob=0.3 mixture share should land near 0.3"
    assert all(1e-4 <= d <= 1e-1 for d in nonzeros), "continuous body must stay within [low, high]"


# --- encoding: descriptors + encode() -----------------------------------------


def test_log_encoding_descriptor_is_box_dim_two() -> None:
    spec = EnvParamSpec(sampling={"type": "loguniform", "low": 1e-4, "high": 1e-1}, encoding={"type": "log"})
    desc = spec.observation_descriptor()
    assert desc.kind == "box" and desc.dim == 2


def test_log_encoding_separates_zero_with_floor_anchor() -> None:
    spec = EnvParamSpec(sampling={"type": "loguniform", "low": 1e-4, "high": 1e-1}, encoding={"type": "log"})
    is_zero, logval = spec.encode_observation(0.0)
    assert is_zero == 1.0
    assert logval == pytest.approx(-4.0)  # log10(floor=low=1e-4)
    is_zero2, logval2 = spec.encode_observation(0.01)
    assert is_zero2 == 0.0
    assert logval2 == pytest.approx(-2.0)


def test_linear_encoding_passes_value_through() -> None:
    spec = EnvParamSpec(values=[1.0, 2.0, 3.0])  # linear inferred
    assert spec.encode_observation(2.0) == [2.0]
    assert spec.observation_descriptor().kind == "box"


def test_categorical_encoding_returns_index_and_discrete_descriptor() -> None:
    spec = EnvParamSpec(values=["a", "b", "c"])  # categorical inferred
    assert spec.encode_observation("b") == 1
    desc = spec.observation_descriptor()
    assert desc.kind == "discrete" and desc.n == 3


def test_asinh_encoding_scales_value() -> None:
    spec = EnvParamSpec(sampling={"type": "loguniform", "low": 1e-4, "high": 1e-1}, encoding={"type": "asinh"})
    assert isinstance(spec.encoding, AsinhEncoding)
    (leaf,) = spec.encode_observation(0.0)
    assert leaf == pytest.approx(0.0)


# --- log encoding: drop_rate == 0 correctness contract ------------------------
#
# log10(0) is -inf, which would poison MeanStdFilter (inf mean/std) and the
# policy net (NaN gradients). The encoder MUST instead emit a finite leaf:
# is_zero=1.0 plus a floor-anchored log term. These tests pin that contract.


def test_log_encoding_zero_drop_rate_is_finite_not_neg_inf() -> None:
    spec = EnvParamSpec(sampling={"type": "loguniform", "low": 1e-4, "high": 1e-1}, encoding={"type": "log"})
    is_zero, logval = spec.encode_observation(0.0)
    assert math.isfinite(is_zero) and math.isfinite(logval), "drop_rate=0 must encode to finite values, never -inf/nan"
    assert is_zero == 1.0, "the is_zero indicator must flag the exact-zero baseline"
    assert logval == pytest.approx(math.log10(1e-4)), "zero anchors at log10(floor=low), not -inf"


def test_log_encoding_zero_uses_categorical_min_positive_floor() -> None:
    """Legacy categorical values [0.0, ...] migrated to log encoding: floor = min positive value."""
    spec = EnvParamSpec(values=[0.0, 0.001, 0.01], encoding={"type": "log"})
    is_zero, logval = spec.encode_observation(0.0)
    assert is_zero == 1.0
    assert math.isfinite(logval)
    assert logval == pytest.approx(math.log10(0.001)), "floor falls back to the smallest positive candidate"


def test_log_encoding_subfloor_and_negative_values_clamp_to_floor() -> None:
    """Defensive: tiny-positive and (spurious) negative rates clamp to the floor, staying finite."""
    spec = EnvParamSpec(
        sampling={"type": "loguniform", "low": 1e-4, "high": 1e-1}, encoding={"type": "log", "floor": 1e-4}
    )
    _, sub_floor = spec.encode_observation(1e-9)
    assert sub_floor == pytest.approx(math.log10(1e-4)), "values below the floor clamp to log10(floor)"
    is_zero_neg, neg = spec.encode_observation(-0.5)
    assert is_zero_neg == 1.0, "non-positive rate is treated as the zero baseline"
    assert math.isfinite(neg) and neg == pytest.approx(math.log10(1e-4))


def test_log_encoding_zero_handled_end_to_end_in_sampler_and_encoder() -> None:
    """Full path: a zero_prob draw of 0.0 round-trips through encode_observation to a finite leaf."""
    spec = EnvParamSpec(
        sampling={"type": "loguniform", "low": 1e-4, "high": 1e-1, "zero_prob": 1.0}, encoding={"type": "log"}
    )
    drawn = EnvParamsSampler({"drop_rate": spec}, seed=0).sample(1)["drop_rate"]
    assert drawn == 0.0, "zero_prob=1.0 must draw the exact-zero baseline"
    is_zero, logval = spec.encode_observation(drawn)
    assert is_zero == 1.0
    assert math.isfinite(logval)
