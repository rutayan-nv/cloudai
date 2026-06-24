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

"""Contract tests for osl domain randomization on the Aiconfigurator workload.

These pin the config-loading half of "works out of the box": the TestDefinition
must accept ``osl`` as a candidate list annotated by ``[env_params.osl]``, keep
``osl`` out of the agent's action space (it is env-sampled, not searched), reject
mis-targeted annotations, and survive a TOML round-trip including an empty
``[env_params.osl]`` table (the uniform marker). A failure here is a job-submit
crash, not a silent degradation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import toml
from pydantic import ValidationError

from cloudai.configurator.env_params import EnvParamSpec
from cloudai.core import TestRun
from cloudai.workloads.aiconfig import AiconfiguratorCmdArgs, AiconfiguratorTestDefinition
from cloudai.workloads.aiconfig.aiconfigurator import Disagg

OSL_CANDIDATES = [256, 512, 1024, 2048]


def _disagg(**overrides: Any) -> Disagg:
    base: dict[str, Any] = dict(
        p_tp=1, p_pp=1, p_dp=1, p_bs=1, p_workers=1,
        d_tp=1, d_pp=1, d_dp=1, d_bs=8, d_workers=2,
    )
    base.update(overrides)
    return Disagg(**base)


def _tdef(
    env_params: dict[str, EnvParamSpec],
    *,
    osl: Any = OSL_CANDIDATES,
    disagg: Disagg | None = None,
) -> AiconfiguratorTestDefinition:
    return AiconfiguratorTestDefinition(
        name="aiconfig_osl_dr",
        description="osl DR",
        test_template_name="Aiconfigurator",
        cmd_args=AiconfiguratorCmdArgs(
            model_name="LLAMA3.1_70B",
            system="h200_sxm",
            isl=4000,
            osl=osl,
            disagg=disagg or _disagg(),
        ),
        agent_metrics=["tokens_per_s_per_gpu"],
        env_params=env_params,
    )


def test_osl_candidate_list_with_uniform_annotation_validates() -> None:
    tdef = _tdef({"osl": EnvParamSpec()})
    assert tdef.cmd_args.osl == OSL_CANDIDATES
    assert isinstance(tdef.env_params["osl"], EnvParamSpec)
    assert tdef.env_params["osl"].weights is None


def test_osl_with_aligned_weights_validates() -> None:
    tdef = _tdef({"osl": EnvParamSpec(weights=[0.4, 0.3, 0.2, 0.1])})
    assert tdef.env_params["osl"].weights == [0.4, 0.3, 0.2, 0.1]


def test_osl_weights_length_must_match_candidates() -> None:
    with pytest.raises(ValidationError):
        _tdef({"osl": EnvParamSpec(weights=[0.5, 0.5])})  # 2 weights vs 4 candidates


def test_env_params_rejects_structured_target() -> None:
    """disagg is a nested model, not a leaf knob; annotating it would silently drop action dims."""
    with pytest.raises(ValidationError, match="leaf cmd_args field"):
        _tdef({"disagg": EnvParamSpec()})


def test_env_params_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError, match="not cmd_args fields"):
        _tdef({"not_a_field": EnvParamSpec()})


def test_osl_excluded_from_action_space_disagg_retained(tmp_path: Path) -> None:
    """osl is env-sampled (never an action dim); the disagg.* lists remain the action space."""
    tdef = _tdef(
        {"osl": EnvParamSpec()},
        disagg=_disagg(p_tp=[1, 2, 4], d_bs=[8, 16, 32], d_workers=[1, 2, 4]),
    )
    tr = TestRun(name="tr", test=tdef, num_nodes=1, nodes=[], output_path=tmp_path / "out")

    keys = set(tr.param_space)
    assert "osl" not in keys, "env-sampled osl must not enter the agent's action space"
    assert "disagg.p_tp" in keys and "disagg.d_bs" in keys and "disagg.d_workers" in keys
    assert tdef.is_dse_job, "disagg.* candidate lists still make this a DSE job"


def test_osl_excluded_does_not_change_action_space_keys(tmp_path: Path) -> None:
    disagg = _disagg(d_bs=[8, 16, 32])
    with_env = _tdef({"osl": EnvParamSpec()}, disagg=disagg)
    without_env = _tdef({}, osl=500, disagg=disagg)
    tr_with = TestRun(name="a", test=with_env, num_nodes=1, nodes=[], output_path=tmp_path / "a")
    tr_without = TestRun(name="b", test=without_env, num_nodes=1, nodes=[], output_path=tmp_path / "b")
    assert set(tr_with.param_space) == set(tr_without.param_space)


def test_toml_roundtrip_preserves_env_params_with_empty_uniform_table() -> None:
    """An empty ``[env_params.osl]`` table must parse to a uniform EnvParamSpec and keep the candidate list."""
    toml_str = """
    name = "aiconfig_osl_dr"
    description = "osl DR"
    test_template_name = "Aiconfigurator"
    agent_metrics = ["tokens_per_s_per_gpu"]
    agent_reward_function = "inverse"

    [cmd_args]
    model_name = "LLAMA3.1_70B"
    system = "h200_sxm"
    backend = "trtllm"
    version = "0.20.0"
    isl = 4000
    osl = [256, 512, 1024, 2048]

      [cmd_args.disagg]
      p_tp = 1
      p_pp = 1
      p_dp = 1
      p_bs = 1
      p_workers = 1
      d_tp = 1
      d_pp = 1
      d_dp = 1
      d_bs = 8
      d_workers = 2

    [env_params.osl]
    """
    parsed = toml.loads(toml_str)
    tdef = AiconfiguratorTestDefinition.model_validate(parsed)

    assert tdef.cmd_args.osl == OSL_CANDIDATES
    assert "osl" in tdef.env_params
    assert isinstance(tdef.env_params["osl"], EnvParamSpec)
    assert tdef.env_params["osl"].weights is None, "empty [env_params.osl] table = uniform sampling"
