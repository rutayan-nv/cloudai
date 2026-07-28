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

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional, Set, Type, TypeAlias, Union

from pydantic import BaseModel

from ..util import flatten_dict
from .action_space import ContinuousSpace
from .system import System

if TYPE_CHECKING:
    from ..models.scenario import ReportConfig
    from ..models.workload import TestDefinition
    from .report_generation_strategy import ReportGenerationStrategy


# Tunable container types that ``param_space`` surfaces as single (un-flattened)
# tunables. Today: list (legacy discrete) and ContinuousSpace (range). Siblings
# of ContinuousSpace (e.g. LogContinuousSpace) get added to this tuple when
# they exist.
_CONTAINER_ACTION_SPACES: tuple[type, ...] = (ContinuousSpace,)


def _collect_action_spaces(model: BaseModel, prefix: str = "") -> dict[str, BaseModel]:
    """Walk ``model``'s typed fields and collect non-list action-space values.

    Recurses into nested Pydantic groups (e.g. ``cmd_args.trainer``) so that
    a deeply-nested space like ``trainer.lr = ContinuousSpace(...)`` is
    surfaced under its dotted key. Stops at any registered action-space type;
    those are leaves regardless of whether they are also Pydantic models.
    """
    out: dict[str, BaseModel] = {}
    for name in model.__class__.model_fields:
        value = getattr(model, name, None)
        full_key = f"{prefix}{name}"
        if isinstance(value, _CONTAINER_ACTION_SPACES):
            out[full_key] = value
        elif isinstance(value, BaseModel):
            out.update(_collect_action_spaces(value, prefix=f"{full_key}."))
    return out


class MetricErrorSentinel:
    """Singleton returned by report strategies on failure; use ``v is METRIC_ERROR`` to detect errors."""

    __slots__ = ()
    _instance: MetricErrorSentinel | None = None

    def __new__(cls) -> MetricErrorSentinel:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "METRIC_ERROR"

    def __float__(self) -> float:
        return -1.0


METRIC_ERROR = MetricErrorSentinel()

MetricValue: TypeAlias = float | MetricErrorSentinel


class TestDependency:
    """
    Represents a dependency for a test.

    Attributes
        test_run (TestRun): TestRun object it depends on.
    """

    __test__ = False

    def __init__(self, test_run: "TestRun") -> None:
        """
        Initialize a TestDependency instance.

        Args:
            test_run (TestRun): TestRun object it depends on.
        """
        self.test_run = test_run


@dataclass
class TestRun:
    __test__ = False

    name: str
    test: TestDefinition
    num_nodes: Union[int, list[int]]
    nodes: List[str]
    exclude_nodes: List[str] = field(default_factory=list)
    output_path: Path = Path("")
    iterations: int = 1
    current_iteration: int = 0
    step: int = 0
    time_limit: Optional[str] = None
    sol: Optional[float] = None
    weight: float = 0.0
    ideal_perf: float = 1.0
    dependencies: dict[str, TestDependency] = field(default_factory=dict)
    pre_test: Optional[TestScenario] = None
    post_test: Optional[TestScenario] = None
    reports: Set[Type[ReportGenerationStrategy]] = field(default_factory=set)
    extra_srun_args: str | None = None
    current_env_params: dict[str, Any] = field(default_factory=dict)

    def __hash__(self) -> int:
        return hash(self.name + self.test.name + str(self.iterations) + str(self.current_iteration))

    def has_more_iterations(self) -> bool:
        """
        Check if the test has more iterations to run.

        Returns
            bool: True if more iterations are pending, False otherwise.
        """
        return self.current_iteration + 1 < self.iterations

    def increment_step(self) -> int:
        """Advance the trial counter and return the new value."""
        self.step += 1
        return self.step

    @property
    def metric_reporter(self) -> Optional[Type[ReportGenerationStrategy]]:
        if not self.reports:
            return None

        if not self.test.agent_metrics:
            return None

        for r in self.reports:
            if all(metric in r.metrics for metric in self.test.agent_metrics):
                return r

        return None

    def get_metric_value(self, system: System, metric: str) -> MetricValue:
        """Resolve a metric name to its current value.

        env-randomized parameters declared on the TestDefinition take
        precedence: when ``metric`` matches a declared env_param key, the
        current trial's sampled value (from ``current_env_params``) is
        returned. This lets observation vectors include trial context (e.g.,
        ``drop_rate``) alongside measured metrics without going through the
        post-run report. Falls back to the report-based metric lookup for
        names that are not env_params.
        """
        if metric in self.test.env_params:
            current = getattr(self, "current_env_params", {}) or {}
            if metric in current:
                return current[metric]

        report = self.metric_reporter
        if report is None:
            return METRIC_ERROR

        return report(system, self).get_metric(metric)

    @property
    def is_dse_job(self) -> bool:
        return self.test.is_dse_job or isinstance(self.num_nodes, list)

    @property
    def nnodes(self) -> int:
        """Type safe getter for num_nodes, should only be used on an unrolled DSE job."""
        if isinstance(self.num_nodes, list):
            raise TypeError("num_nodes is a list, cannot be used as a scalar.")
        return self.num_nodes

    @property
    def param_space(self) -> dict[str, Any]:
        cmd_args_dict = flatten_dict(self.test.cmd_args.model_dump())

        space_overrides = _collect_action_spaces(self.test.cmd_args)
        for key in space_overrides:
            for k in [k for k in cmd_args_dict if k == key or k.startswith(f"{key}.")]:
                cmd_args_dict.pop(k)
        cmd_args_dict.update(space_overrides)

        extra_env_vars_dict = self.test.extra_env_vars

        action_space: dict[str, Any] = {
            **{
                key: value
                for key, value in cmd_args_dict.items()
                if (isinstance(value, (list,) + _CONTAINER_ACTION_SPACES))
                and not self.test.is_dse_excluded_arg(key)
            },
            **{f"extra_env_vars.{key}": value for key, value in extra_env_vars_dict.items() if isinstance(value, list)},
        }
        if isinstance(self.num_nodes, list):
            action_space["NUM_NODES"] = self.num_nodes

        return action_space

    @property
    def all_combinations(self) -> list[dict[str, Any]]:
        if not self.is_dse_job:
            return []

        param_space: dict[str, Any] = self.param_space
        if not param_space:
            return []

        non_enumerable = [
            key for key, value in param_space.items() if isinstance(value, _CONTAINER_ACTION_SPACES)
        ]
        if non_enumerable:
            raise TypeError(
                f"all_combinations cannot enumerate continuous action spaces: {sorted(non_enumerable)}. "
                "Grid-search and exhaustive sweeps require list-typed (discrete) tunables; use an "
                "RL agent (PPO/DQN) for continuous action spaces."
            )

        parameter_values: list[Any] = []
        for _, values in param_space.items():
            parameter_values.append(values)
        action_combinations = list(itertools.product(*parameter_values))

        keys = list(param_space.keys())
        all_combinations = [dict(zip(keys, combination, strict=True)) for combination in action_combinations]

        return all_combinations

    def apply_params_set(self, action: dict[str, Any]) -> "TestRun":
        tdef = self.test.model_copy(deep=True)
        for key, value in action.items():
            if key.startswith("extra_env_vars."):
                tdef.extra_env_vars[key[len("extra_env_vars.") :]] = value
            else:
                attrs = key.split(".")
                obj = tdef.cmd_args
                for attr in attrs[:-1]:
                    obj = obj[attr] if isinstance(obj, dict) else getattr(obj, attr)
                if isinstance(obj, dict):
                    obj[attrs[-1]] = value
                else:
                    setattr(obj, attrs[-1], value)

        type(tdef)(**tdef.model_dump())  # trigger validation

        new_tr = copy.deepcopy(self)
        new_tr.test = tdef
        if "NUM_NODES" in action:
            new_tr.num_nodes = action["NUM_NODES"]
        return new_tr


@dataclass
class TestScenario:
    """
    Represents a test scenario, comprising a set of tests.

    Attributes
        name (str): Unique name of the test scenario.
        tests (List[Test]): Tests in the scenario.
        job_status_check (bool): Flag indicating whether to check the job status or not.
        reports (dict[str, ReportConfig] | None): Report configurations for the scenario.
    """

    __test__ = False

    name: str
    test_runs: list[TestRun]
    job_status_check: bool = True
    reports: dict[str, ReportConfig] = field(default_factory=dict)

    def __repr__(self) -> str:
        """
        Return a string representation of the TestScenario instance.

        Returns
            str: String representation of the test scenario.
        """
        test_names = ", ".join([tr.test.name for tr in self.test_runs])
        return f"TestScenario(name={self.name}, tests=[{test_names}])"

    def pretty_print(self) -> str:
        """Print each test in the scenario along with its section name, description, and visualized dependencies."""
        s = f"Test Scenario: {self.name}\n"
        for tr in self.test_runs:
            s += f"\nSection Name: {tr.name}\n"
            s += f"  Test Name: {tr.test.name}\n"
            s += f"  Description: {tr.test.description}\n"
            if tr.dependencies:
                for dep_type, dependency in tr.dependencies.items():
                    if dependency:
                        s += f"  {dep_type.replace('_', ' ').title()}: {dependency.test_run.name}"
            else:
                s += "  No dependencies"
        return s
