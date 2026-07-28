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
Cache-prefill loader for ``CloudAIGymEnv``.

Reads a prior run's ``trajectory.csv`` (and optional sibling ``env.csv``) into
``TrajectoryEntry`` objects so a new ``CloudAIGymEnv`` can pre-populate its
in-memory cache and short-circuit cluster execution on repeated
``(action, env_params)`` trials.

Distinct from the BC-warm-start loader at
``domain_randomization/utils/configurator/trajectory_loader.py``: that loader
strips ``env_params`` and reshapes rows for a behavioral-cloning corpus; this
one preserves full trial identity for the env-level cache.
"""

import ast
import csv
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from .cloudai_gym import TrajectoryEntry


def load_trajectory_with_env(
    traj_csv: Path,
    env_csv: Optional[Path] = None,
) -> list[TrajectoryEntry]:
    """
    Load a prior run's trajectory + env-params into ``TrajectoryEntry`` list.

    Args:
        traj_csv: Path to a ``trajectory.csv`` written by ``CloudAIGymEnv``
            (header ``step,action,reward,observation``).
        env_csv: Optional path to an ``env.csv`` written by ``CsvSink``
            (header ``step,env``). If ``None``, this function looks for
            ``env.csv`` in the same directory as ``traj_csv``; a missing
            sibling is silently treated as "no env_params recorded" (correct
            for workloads with no ``[env_params.*]`` block). An explicit
            path that does not exist raises ``FileNotFoundError``.

    Returns:
        A list of ``TrajectoryEntry`` ordered as in ``traj_csv``. Rows with
        non-positive reward (sentinel ``-1.0`` constraint failures, ``0.0``
        no-ops) and rows whose ``action`` is not a parseable mapping are
        skipped. Returns ``[]`` when nothing survives filtering — empty
        prefill is not an error for the cache use case.

    Raises:
        FileNotFoundError: ``traj_csv`` does not exist, or an explicitly
            supplied ``env_csv`` does not exist.
    """
    traj_path = Path(traj_csv)
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_csv}")

    env_by_step = _load_env_by_step(traj_path, env_csv)

    entries: list[TrajectoryEntry] = []
    with traj_path.open() as fh:
        for row in csv.DictReader(fh):
            entry = _parse_trajectory_row(row, env_by_step)
            if entry is not None:
                entries.append(entry)
    return entries


def _load_env_by_step(traj_path: Path, env_csv: Optional[Path]) -> Dict[int, Dict[str, Any]]:
    """
    Resolve env.csv path and parse it into a step→sample dict.

    Auto-discovery (``env_csv is None``) returns an empty mapping when no
    sibling exists; an explicit-but-missing ``env_csv`` is a hard error.
    """
    if env_csv is None:
        candidate = traj_path.parent / "env.csv"
        if not candidate.exists():
            return {}
        env_path = candidate
    else:
        env_path = Path(env_csv)
        if not env_path.exists():
            raise FileNotFoundError(f"env.csv path was supplied but does not exist: {env_csv}")

    by_step: Dict[int, Dict[str, Any]] = {}
    with env_path.open() as fh:
        for row in csv.DictReader(fh):
            try:
                step = int(row["step"])
                sample = ast.literal_eval(row["env"])
            except (KeyError, TypeError, ValueError, SyntaxError):
                logging.debug("Skipping malformed env.csv row: %r", row)
                continue
            if isinstance(sample, dict):
                by_step[step] = sample
    return by_step


def _parse_trajectory_row(
    row: Dict[str, str],
    env_by_step: Dict[int, Dict[str, Any]],
) -> Optional[TrajectoryEntry]:
    """Parse one trajectory.csv row; return ``None`` to drop it."""
    try:
        reward = float(row["reward"])
    except (KeyError, TypeError, ValueError):
        return None
    if reward <= 0.0:
        return None

    try:
        action = ast.literal_eval(row["action"])
    except (KeyError, ValueError, SyntaxError):
        return None
    if not isinstance(action, dict):
        return None

    try:
        observation = ast.literal_eval(row.get("observation", "[]"))
    except (ValueError, SyntaxError):
        observation = []
    if not isinstance(observation, list):
        observation = []

    try:
        step = int(row["step"])
    except (KeyError, TypeError, ValueError):
        return None

    return TrajectoryEntry(
        step=step,
        action=action,
        reward=reward,
        observation=observation,
        env_params=env_by_step.get(step, {}),
    )
