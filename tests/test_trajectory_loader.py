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

import csv
from pathlib import Path

import pytest

from cloudai.configurator import TrajectoryEntry
from cloudai.configurator.trajectory_loader import load_trajectory_with_env


def _write_trajectory_csv(path: Path, rows: list[tuple[int, dict, float, list]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "action", "reward", "observation"])
        for step, action, reward, observation in rows:
            writer.writerow([step, action, reward, observation])


def _write_env_csv(path: Path, rows: list[tuple[int, dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "env"])
        for step, env in rows:
            writer.writerow([step, env])


def test_load_trajectory_with_env_round_trip(tmp_path: Path) -> None:
    """Three rows in, three TrajectoryEntry out, each carrying its env_params."""
    traj = tmp_path / "trajectory.csv"
    env = tmp_path / "env.csv"
    _write_trajectory_csv(
        traj,
        [
            (1, {"prt_ooo_threshold": 17}, 0.83, [0.83, 0.0]),
            (2, {"prt_ooo_threshold": 100}, 0.55, [0.55, 0.001]),
            (3, {"prt_ooo_threshold": 200}, 0.27, [0.27, 0.01]),
        ],
    )
    _write_env_csv(
        env,
        [
            (1, {"drop_rate": 0.0}),
            (2, {"drop_rate": 0.001}),
            (3, {"drop_rate": 0.01}),
        ],
    )

    entries = load_trajectory_with_env(traj)

    assert len(entries) == 3
    assert all(isinstance(e, TrajectoryEntry) for e in entries)
    assert entries[0].step == 1
    assert entries[0].action == {"prt_ooo_threshold": 17}
    assert entries[0].reward == pytest.approx(0.83)
    assert entries[0].observation == [0.83, 0.0]
    assert entries[0].env_params == {"drop_rate": 0.0}
    assert entries[2].env_params == {"drop_rate": 0.01}


def test_load_preserves_int_action_type(tmp_path: Path) -> None:
    """Cache key match is type-strict (`type(left) is not type(right)`).

    A loaded action whose value is an int must remain an int, not a float —
    otherwise prefilled entries silently miss against ints emitted by the
    quantized continuous-action path.
    """
    traj = tmp_path / "trajectory.csv"
    _write_trajectory_csv(traj, [(1, {"prt_ooo_threshold": 47}, 0.5, [0.5])])

    entries = load_trajectory_with_env(traj)

    assert len(entries) == 1
    loaded_value = entries[0].action["prt_ooo_threshold"]
    assert isinstance(loaded_value, int)
    assert not isinstance(loaded_value, bool)


def test_load_skips_non_positive_rewards(tmp_path: Path) -> None:
    """Failed/constraint-violating trials must re-execute, not hit a stale -1.0."""
    traj = tmp_path / "trajectory.csv"
    _write_trajectory_csv(
        traj,
        [
            (1, {"x": 1}, 0.5, [0.5]),
            (2, {"x": 2}, -1.0, [-1.0]),
            (3, {"x": 3}, 0.0, [0.0]),
            (4, {"x": 4}, 0.7, [0.7]),
        ],
    )

    entries = load_trajectory_with_env(traj)

    assert [e.step for e in entries] == [1, 4]


def test_load_missing_traj_csv_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_trajectory_with_env(tmp_path / "does_not_exist.csv")


def test_load_without_env_csv_yields_empty_env_params(tmp_path: Path) -> None:
    """Workloads with no [env_params.*] block produce no env.csv. Prefill must still work."""
    traj = tmp_path / "trajectory.csv"
    _write_trajectory_csv(traj, [(1, {"x": 1}, 0.5, [0.5])])

    entries = load_trajectory_with_env(traj)

    assert len(entries) == 1
    assert entries[0].env_params == {}


def test_load_explicit_env_csv_path(tmp_path: Path) -> None:
    """User can point at an env.csv that lives outside the trajectory's directory."""
    traj = tmp_path / "run/trajectory.csv"
    env = tmp_path / "elsewhere/env.csv"
    _write_trajectory_csv(traj, [(1, {"x": 1}, 0.5, [0.5])])
    _write_env_csv(env, [(1, {"drop_rate": 0.42})])

    entries = load_trajectory_with_env(traj, env_csv=env)

    assert entries[0].env_params == {"drop_rate": 0.42}


def test_load_explicit_env_csv_missing_raises(tmp_path: Path) -> None:
    """An explicit env_csv that does not exist is a hard error (user typo'd the path)."""
    traj = tmp_path / "trajectory.csv"
    _write_trajectory_csv(traj, [(1, {"x": 1}, 0.5, [0.5])])

    with pytest.raises(FileNotFoundError):
        load_trajectory_with_env(traj, env_csv=tmp_path / "missing.csv")


def test_load_env_csv_partial_coverage(tmp_path: Path) -> None:
    """Trajectory rows without a matching env.csv row land with empty env_params."""
    traj = tmp_path / "trajectory.csv"
    env = tmp_path / "env.csv"
    _write_trajectory_csv(
        traj,
        [
            (1, {"x": 1}, 0.5, [0.5]),
            (2, {"x": 2}, 0.6, [0.6]),
            (3, {"x": 3}, 0.7, [0.7]),
        ],
    )
    _write_env_csv(env, [(1, {"drop_rate": 0.0}), (3, {"drop_rate": 0.01})])

    entries = load_trajectory_with_env(traj)

    by_step = {e.step: e for e in entries}
    assert by_step[1].env_params == {"drop_rate": 0.0}
    assert by_step[2].env_params == {}
    assert by_step[3].env_params == {"drop_rate": 0.01}


def test_load_skips_malformed_action_row(tmp_path: Path) -> None:
    """A row with a non-dict action is dropped (logged), not raised."""
    traj = tmp_path / "trajectory.csv"
    with traj.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "action", "reward", "observation"])
        writer.writerow([1, "not a dict", 0.5, [0.5]])
        writer.writerow([2, {"x": 2}, 0.6, [0.6]])

    entries = load_trajectory_with_env(traj)

    assert [e.step for e in entries] == [2]


def test_load_empty_traj_csv_returns_empty_list(tmp_path: Path) -> None:
    """No rows after filtering is not an error (BC fork raised; cache prefill must not)."""
    traj = tmp_path / "trajectory.csv"
    _write_trajectory_csv(traj, [(1, {"x": 1}, -1.0, [-1.0])])

    entries = load_trajectory_with_env(traj)

    assert entries == []


# ---------------------------------------------------------------------------
# Stage 0 backward-compat: the observation column may now be
# [<metrics...>, <context...>]. The warm-start reconstruction must peel the
# leading metrics back off so entry.observation is context-only again, while
# old-format (context-only) files are left untouched.
# ---------------------------------------------------------------------------


def test_load_new_format_strips_leading_metrics_from_observation(tmp_path: Path) -> None:
    """New-format rows (observation = [<metric>, <context>]) split back into metrics + context.

    With num_metrics=1 and num_observation=1, an observation cell of length 2
    is the new format, so the leading metric is peeled off and entry.observation
    (the policy-facing vector) is context-only again.
    """
    traj = tmp_path / "trajectory.csv"
    _write_trajectory_csv(traj, [(1, {"prt_ooo_threshold": 17}, 0.83, [11.5, 0.003])])

    entries = load_trajectory_with_env(traj, num_metrics=1, num_observation=1)

    assert len(entries) == 1
    assert entries[0].observation == [0.003], "leading metric must be stripped from the policy observation"
    assert entries[0].metrics == [11.5], "stripped metric value is preserved on the entry"


def test_load_old_format_observation_unchanged(tmp_path: Path) -> None:
    """Old-format rows (observation length == context length) are left intact.

    A run written before Stage 0 has observation = [context...]. The length
    discriminator (new format iff len == num_metrics + num_observation) must not
    misread a genuine multi-value context and strip a real observation value.
    """
    traj = tmp_path / "trajectory.csv"
    _write_trajectory_csv(traj, [(1, {"x": 1}, 0.83, [0.83, 0.0])])

    entries = load_trajectory_with_env(traj, num_metrics=1, num_observation=2)

    assert entries[0].observation == [0.83, 0.0], "old-format context observation must be preserved"
    assert entries[0].metrics == []


def test_load_defaults_do_not_strip_observation(tmp_path: Path) -> None:
    """Existing callers (no num_metrics/num_observation) keep the legacy behavior: no stripping."""
    traj = tmp_path / "trajectory.csv"
    _write_trajectory_csv(traj, [(1, {"x": 1}, 0.5, [11.5, 0.003])])

    entries = load_trajectory_with_env(traj)

    assert entries[0].observation == [11.5, 0.003]
    assert entries[0].metrics == []
