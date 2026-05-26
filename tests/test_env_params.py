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

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest
from pydantic import ValidationError

from cloudai.configurator.env_params import (
    CsvSink,
    EnvParamsObserver,
    EnvParamSpec,
    EnvParamsSampler,
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
