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

"""
End-to-end test of cache prefill with the simulator entry point mocked.

The mock target is ``cloudai.util.command_shell.CommandShell.execute`` —
exactly the boundary at which a real run launches the workload bash script
(and, for nvnsim workloads, the nvnsim binary). Mocking via
``unittest.mock.patch`` (no fake script file required) keeps the test
simulator-free while exercising the *full* CloudAI launch pipeline:
``CloudAIGymEnv.step`` → ``StandaloneRunner._submit_test`` →
``CommandShell.execute`` → ``StandaloneSystem.is_job_completed`` →
``BaseRunner.handle_job_completion``.

The cache prefill contract under test:
    For each (action, env_params) pair already present in the prior
    run's ``trajectory.csv`` + ``env.csv``, ``CommandShell.execute`` must
    NOT be called (cluster compute is saved). Novel pairs must invoke
    ``CommandShell.execute`` exactly once.
"""

from __future__ import annotations

import csv
from collections.abc import Generator
from pathlib import Path
from typing import cast
from unittest.mock import Mock, patch

import pytest

from cloudai.configurator import CloudAIGymEnv
from cloudai.core import CommandGenStrategy, Registry, RewardOverrides, Runner, TestRun, TestScenario
from cloudai.models.workload import CmdArgs, TestDefinition
from cloudai.systems.standalone import StandaloneSystem


class _StubCmdArgs(CmdArgs):
    """Single-knob stub command-args used as the action in this e2e test."""

    seconds: int = 5


class _StubTestDefinition(TestDefinition):
    """Stub workload TestDefinition; never reaches a real simulator."""

    cmd_args: _StubCmdArgs


class _StubStandaloneCmdGen(CommandGenStrategy):
    """Minimal Standalone command-gen: returns ``echo seconds=<n>``.

    The test does not care about the contents of the command; only that it
    is produced and handed to ``CommandShell.execute`` for every cache
    miss. Using ``echo`` keeps the bash invocation harmless even if the
    mock ever leaks.
    """

    def gen_exec_command(self) -> str:
        td = cast(_StubTestDefinition, self.test_run.test)
        return f"echo seconds={td.cmd_args.seconds}"

    def store_test_run(self) -> None:
        return


@pytest.fixture
def stub_standalone_strategy_registered() -> Generator[None, None, None]:
    """Register the (StandaloneSystem, _StubTestDefinition) → _StubStandaloneCmdGen mapping for one test."""
    registry = Registry()
    registry.add_command_gen_strategy(StandaloneSystem, _StubTestDefinition, _StubStandaloneCmdGen)
    yield
    del registry.command_gen_strategies_map[(StandaloneSystem, _StubTestDefinition)]


def _write_prior_trajectory_csv(path: Path, rows: list[tuple[int, dict, float, list]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "action", "reward", "observation"])
        for step, action, reward, obs in rows:
            w.writerow([step, action, reward, obs])


def _write_prior_env_csv(path: Path, rows: list[tuple[int, dict]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "env"])
        for step, env in rows:
            w.writerow([step, env])


@pytest.fixture
def primed_test_run(tmp_path: Path) -> tuple[TestRun, Path]:
    """Build a real ``_StubTestDefinition`` whose cache is primed by 2 prior trials."""
    prior_dir = tmp_path / "prior_run"
    traj = prior_dir / "trajectory.csv"
    _write_prior_trajectory_csv(
        traj,
        [
            (1, {"seconds": 5}, 0.7, [0.7]),
            (2, {"seconds": 7}, 0.8, [0.8]),
        ],
    )
    _write_prior_env_csv(prior_dir / "env.csv", [(1, {"drop_rate": 0.0}), (2, {"drop_rate": 0.0})])

    tdef = _StubTestDefinition(
        name="prefill-e2e",
        description="Cache-prefill e2e against a stub Standalone workload.",
        test_template_name="StubTest",
        cmd_args=_StubCmdArgs(seconds=5),
        agent_metrics=["default"],
        cache_warm_start_path=traj,
    )
    test_run = TestRun(
        name="stub_run",
        test=tdef,
        num_nodes=1,
        nodes=[],
        output_path=tmp_path / "out" / "stub_run",
    )
    return test_run, traj


def _fake_completed_process() -> Mock:
    proc = Mock()
    proc.pid = 99999
    proc.poll.return_value = 0
    proc.returncode = 0
    proc.communicate.return_value = ("", "")
    return proc


@pytest.mark.usefixtures("stub_standalone_strategy_registered")
def test_e2e_primed_trials_skip_simulator_entry_point(
    primed_test_run: tuple[TestRun, Path],
    standalone_system: StandaloneSystem,
) -> None:
    """Primed actions short-circuit; novel actions invoke ``CommandShell.execute`` once each.

    Drives a 3-trial sequence (primed, novel, primed) through the *real*
    StandaloneRunner. The only mocks are at the simulator-launch boundary
    and the job-completion poll — everything else (cmd-gen, output-path
    creation, trajectory.csv writing, cache lookup) is real.
    """
    test_run, _traj = primed_test_run
    test_scenario = TestScenario(name="prefill_e2e_scenario", test_runs=[test_run])
    runner = Runner(mode="run", system=standalone_system, test_scenario=test_scenario)

    execute_target = "cloudai.util.command_shell.CommandShell.execute"
    with (
        patch(execute_target, return_value=_fake_completed_process()) as mock_execute,
        patch.object(StandaloneSystem, "is_job_running", return_value=False),
        patch("time.sleep", return_value=None),
        # Reward inputs come from agent_metrics — mock the metric-resolution
        # boundary (TestRun.get_metric_value) so the cache-miss row computes
        # ``inverse_reward([0.5]) = 2.0``. The observation path is independent;
        # it doesn't need a stub here because no agent_observation is set on
        # the stub workload.
        patch.object(TestRun, "get_metric_value", return_value=0.5),
    ):
        env = CloudAIGymEnv(test_run=test_run, runner=runner.runner, rewards=RewardOverrides())

        assert len(env.trajectory.get(0, [])) == 2, "prefill must load both prior rows before the agent loop starts"

        rewards: list[float] = []
        for action in ({"seconds": 5}, {"seconds": 99}, {"seconds": 7}):
            env.test_run.current_env_params = {"drop_rate": 0.0}
            env.reset()
            _obs, reward, _done, _info = env.step(action)
            rewards.append(reward)

    assert mock_execute.call_count == 1, (
        f"only the novel trial (seconds=99) should invoke the simulator entry point; "
        f"got {mock_execute.call_count} calls. Primed trials must short-circuit."
    )
    invoked_cmd = mock_execute.call_args.args[0]
    assert "seconds=99" in invoked_cmd, f"the one entry-point call must be for the novel action; got: {invoked_cmd}"

    assert rewards[0] == pytest.approx(0.7), "trial 1 must replay the prior reward verbatim"
    assert rewards[2] == pytest.approx(0.8), "trial 3 must replay the prior reward verbatim"
    assert rewards[1] == pytest.approx(2.0), "trial 2 (cache miss) must compute via inverse_reward([0.5]) = 2.0"


@pytest.mark.usefixtures("stub_standalone_strategy_registered")
def test_e2e_env_params_mismatch_forces_simulator_invocation(
    primed_test_run: tuple[TestRun, Path],
    standalone_system: StandaloneSystem,
) -> None:
    """Domain-randomization soundness: same action under a *different* env_params must MISS.

    Primed entries have ``drop_rate=0.0``. A trial with ``seconds=5`` under
    ``drop_rate=0.01`` shares the action but not the env_params, so the
    cache key must miss and ``CommandShell.execute`` must run the workload.
    Regression test for the env-params-aware cache key under prefill.
    """
    test_run, _traj = primed_test_run
    test_scenario = TestScenario(name="env_params_mismatch_scenario", test_runs=[test_run])
    runner = Runner(mode="run", system=standalone_system, test_scenario=test_scenario)

    execute_target = "cloudai.util.command_shell.CommandShell.execute"
    with (
        patch(execute_target, return_value=_fake_completed_process()) as mock_execute,
        patch.object(StandaloneSystem, "is_job_running", return_value=False),
        patch("time.sleep", return_value=None),
        patch.object(CloudAIGymEnv, "get_observation", return_value=[0.5]),
    ):
        env = CloudAIGymEnv(test_run=test_run, runner=runner.runner, rewards=RewardOverrides())
        env.test_run.current_env_params = {"drop_rate": 0.01}
        env.reset()
        env.step({"seconds": 5})

    assert mock_execute.call_count == 1, (
        "same action under a different env_params must miss the prefilled cache and invoke the simulator"
    )
    assert "seconds=5" in mock_execute.call_args.args[0]


@pytest.mark.usefixtures("stub_standalone_strategy_registered")
def test_e2e_trajectory_csv_records_all_trials_including_cache_hits(
    primed_test_run: tuple[TestRun, Path],
    standalone_system: StandaloneSystem,
) -> None:
    """Cache hits must still append a row; the visible step list reflects the full episode."""
    test_run, _traj = primed_test_run
    test_scenario = TestScenario(name="record_all_scenario", test_runs=[test_run])
    runner = Runner(mode="run", system=standalone_system, test_scenario=test_scenario)

    with (
        patch("cloudai.util.command_shell.CommandShell.execute", return_value=_fake_completed_process()),
        patch.object(StandaloneSystem, "is_job_running", return_value=False),
        patch("time.sleep", return_value=None),
        patch.object(CloudAIGymEnv, "get_observation", return_value=[0.5]),
    ):
        env = CloudAIGymEnv(test_run=test_run, runner=runner.runner, rewards=RewardOverrides())
        for action in ({"seconds": 5}, {"seconds": 99}, {"seconds": 7}):
            env.test_run.current_env_params = {"drop_rate": 0.0}
            env.reset()
            env.step(action)

        csv_path = env.trajectory_file_path

    assert csv_path.exists(), "trajectory.csv must be written under runner.scenario_root"
    lines = csv_path.read_text().strip().splitlines()
    assert lines[0] == "step,action,reward,observation", "header must be the canonical 4-column form"
    assert len(lines) == 1 + 3, "header + 3 trial rows (cache hits and misses both write rows)"
