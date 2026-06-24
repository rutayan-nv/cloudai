# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""End-to-end no-op guard for osl domain randomization on the Aiconfigurator workload.

The Aiconfigurator predictor takes ``--osl`` as ``type=int`` (see
``runtime/simple_predictor.py``). Domain randomization declares ``osl`` as a
candidate list in ``cmd_args`` plus an ``[env_params.osl]`` annotation, so the
*static* cmd_args value is a list. If the per-trial sample is not overlaid onto
cmd_args before command generation, the strategy stringifies the whole list and
emits ``--osl '[256, 512, ...]'`` -> ``int(...)`` raises at runtime.

These tests pin the full chain (sample -> CloudAIGymEnv overlay -> command gen)
so the sampled scalar reaches the emitted predictor command and the list never
leaks. ``osl`` is the env-randomized traffic knob; ``disagg.*`` stays the action
space (here scalar, so no action dimensions are involved).
"""

from __future__ import annotations

import importlib.util
import random
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import cloudai.workloads.aiconfig.standalone_command_gen_strategy as _scg
from cloudai.configurator import CloudAIGymEnv
from cloudai.configurator.env_params import EnvParamSpec
from cloudai.core import BaseRunner, RewardOverrides, TestRun, TestScenario
from cloudai.systems.standalone import StandaloneSystem
from cloudai.workloads.aiconfig import (
    AiconfiguratorCmdArgs,
    AiconfiguratorStandaloneCommandGenStrategy,
    AiconfiguratorTestDefinition,
)
from cloudai.workloads.aiconfig.aiconfigurator import Disagg

OSL_CANDIDATES = [256, 512, 1024, 2048]

# The exact predictor entrypoint the strategy invokes (see standalone_command_gen_strategy.py).
PREDICTOR_SCRIPT = Path(_scg.__file__).with_name("runtime") / "simple_predictor.py"


def _scalar_disagg() -> Disagg:
    """A fully-scalar disagg config: no action-space dimensions, so the trial is driven by env_params alone."""
    return Disagg(
        p_tp=1, p_pp=1, p_dp=1, p_bs=1, p_workers=1,
        d_tp=1, d_pp=1, d_dp=1, d_bs=8, d_workers=2,
        prefill_correction_scale=1.0, decode_correction_scale=1.0,
    )


def _aiconfig_tdef() -> AiconfiguratorTestDefinition:
    return AiconfiguratorTestDefinition(
        name="aiconfig_osl_dr",
        description="osl domain randomization",
        test_template_name="Aiconfigurator",
        cmd_args=AiconfiguratorCmdArgs(
            model_name="LLAMA3.1_70B",
            system="h200_sxm",
            backend="trtllm",
            version="0.20.0",
            isl=4000,
            osl=OSL_CANDIDATES,
            disagg=_scalar_disagg(),
        ),
        agent_metrics=["tokens_per_s_per_gpu"],
        agent_config={"random_seed": 42},
        env_params={"osl": EnvParamSpec()},
    )


def _osl_token(script: str) -> str:
    """Return the single token passed to ``--osl`` in the generated bash command."""
    match = re.search(r"--osl\s+(\S+)", script)
    assert match is not None, f"no --osl flag in generated command:\n{script}"
    return match.group(1)


def test_sampled_osl_overlay_reaches_predictor_command(tmp_path: Path, standalone_system: StandaloneSystem) -> None:
    """Driving one CloudAIGymEnv.step() must overlay the sampled osl so the emitted command is a valid int."""
    tdef = _aiconfig_tdef()
    test_run = TestRun(
        name="aiconfig_osl_dr_tr",
        test=tdef,
        num_nodes=1,
        nodes=[],
        output_path=tmp_path / "out" / "aiconfig_osl_dr_tr" / "0",
    )

    runner = MagicMock(spec=BaseRunner)
    runner.scenario_root = tmp_path / "scenario"
    runner.system = MagicMock()
    runner.test_scenario = TestScenario(name="aiconfig_osl_dr_scenario", test_runs=[test_run])
    runner.jobs, runner.testrun_to_job_map, runner.shutting_down = {}, {}, False
    runner.get_job_output_path.return_value = test_run.output_path

    env = CloudAIGymEnv(test_run=test_run, runner=runner, rewards=RewardOverrides())

    # Sampler is seeded f"{random_seed}:{name}:{trial}"; the first step is trial 1.
    expected = random.Random("42:osl:1").choice(OSL_CANDIDATES)
    assert expected in OSL_CANDIDATES

    with patch.object(env, "get_observation", side_effect=lambda _action: [1.0]):
        env.test_run.step = 0
        env.step({})

    # The run the runner would have executed carries the overlaid (scalar) osl.
    overlaid_tr = runner.test_scenario.test_runs[0]
    assert overlaid_tr.test.cmd_args.osl == expected, (
        "step() must overlay the sampled osl onto cmd_args before the workload runs; "
        f"got {overlaid_tr.test.cmd_args.osl!r}, expected the scalar {expected}."
    )

    strategy = AiconfiguratorStandaloneCommandGenStrategy(standalone_system, overlaid_tr)
    strategy.gen_exec_command()
    script = (overlaid_tr.output_path.resolve() / "run_simple_predictor.sh").read_text(encoding="utf-8")

    token = _osl_token(script)
    assert token == str(expected), f"--osl should carry the sampled scalar {expected}, got {token!r}"
    assert int(token) == expected, "predictor parses --osl with type=int; the token must be int-parseable"
    assert "[" not in script.split("--osl", 1)[1][:40], "the osl candidate list must never leak into the command"


def test_osl_candidate_list_leaks_without_overlay(tmp_path: Path, standalone_system: StandaloneSystem) -> None:
    """Negative control: with no overlay the static list leaks and is not int-parseable (the failure we prevent)."""
    tdef = _aiconfig_tdef()
    test_run = TestRun(
        name="aiconfig_osl_noop_tr",
        test=tdef,
        num_nodes=1,
        nodes=[],
        output_path=tmp_path / "noop" / "aiconfig_osl_noop_tr" / "0",
    )

    strategy = AiconfiguratorStandaloneCommandGenStrategy(standalone_system, test_run)
    strategy.gen_exec_command()
    script = (test_run.output_path.resolve() / "run_simple_predictor.sh").read_text(encoding="utf-8")

    token = _osl_token(script)
    try:
        int(token)
        parsed = True
    except ValueError:
        parsed = False
    assert not parsed, (
        "Without the per-trial overlay the candidate list reaches the command and breaks the int parse; "
        "this is exactly the runtime failure the overlay prevents."
    )


# --- Predictor CLI canary: the other end of the contract --------------------------------------
#
# The injection tests prove the strategy *emits* ``--osl <int>``. These prove the shipped predictor
# *accepts* ``--osl <int>`` and *rejects* a list literal. If aiconfigurator/predictor drift renames
# ``--osl`` or changes its type away from int, these fail loudly instead of at job runtime.


def _load_predictor_module():
    spec = importlib.util.spec_from_file_location("_aiconfig_predictor_canary", PREDICTOR_SCRIPT)
    assert spec is not None and spec.loader is not None, f"cannot load predictor at {PREDICTOR_SCRIPT}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _predictor_argv(osl: str) -> list[str]:
    return [
        "simple_predictor.py",
        "--mode", "disagg",
        "--model-name", "LLAMA3.1_70B",
        "--system", "h200_sxm",
        "--isl", "4000",
        "--osl", osl,
        "--output", "out.json",
    ]


def test_predictor_entrypoint_exists() -> None:
    assert PREDICTOR_SCRIPT.is_file(), f"predictor entrypoint missing at {PREDICTOR_SCRIPT}"


def test_predictor_accepts_scalar_int_osl(monkeypatch: pytest.MonkeyPatch) -> None:
    predictor = _load_predictor_module()
    monkeypatch.setattr(sys, "argv", _predictor_argv("1024"))
    ns = predictor.parse_args()
    assert ns.osl == 1024 and isinstance(ns.osl, int), "predictor must parse the emitted scalar --osl as int"


def test_predictor_rejects_list_literal_osl(monkeypatch: pytest.MonkeyPatch) -> None:
    predictor = _load_predictor_module()
    monkeypatch.setattr(sys, "argv", _predictor_argv("[256, 512, 1024, 2048]"))
    with pytest.raises(SystemExit):
        predictor.parse_args()
