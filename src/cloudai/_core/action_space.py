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

"""Continuous action space for cmd_arg tunables.

Background. cloudai's TOML expresses tunable cmd_args two ways:

* ``foo = [a, b, c]`` — a discrete list. Each agent imposes its own semantic
  (RL: unordered Discrete; BO: ordinal RangeParameter with auto log-detect).
* ``foo = { low = 0, high = 200, dtype = "int" }`` — a continuous range
  (this module). The agent's policy operates over a real interval; for
  ``dtype="int"`` the workload rounds at the command boundary.

The continuous form lands here as a single Pydantic class because it has a
distinct field shape (``low``/``high``) that no other tunable form has.
``TestRun.param_space`` surfaces :class:`ContinuousSpace` instances as single
tunables (rather than letting ``flatten_dict`` explode them into
``param.low`` / ``param.high`` scalars). Adapters consuming ``param_space``
dispatch on it — RL → ``gym.Box``; BO → ``RangeParameter``; etc. — without
further plumbing in cloudai core.

When a second action-space shape arrives (e.g. ``LogContinuousSpace``,
``OrdinalSpace`` with metadata), promote to a discriminated union or sibling
classes at that point. Today it's one class; that's all the surface area we
need.

Lives in ``_core`` (not ``configurator``) because :class:`TestRun.param_space`
is the consumer and ``configurator`` already imports from ``_core``; placing
it in ``configurator`` would create a circular import via that package's
``__init__``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator


class ContinuousSpace(BaseModel):
    """Continuous Box action space over ``[low, high]``.

    ``dtype="float"`` exposes a real-valued action; ``dtype="int"`` quantizes
    the policy's continuous output to an integer at the command boundary
    (the policy still trains on floats — quantization is an env property,
    not a learning bug).
    """

    model_config = ConfigDict(frozen=True)

    low: float
    high: float
    dtype: Literal["float", "int"] = "float"

    @model_validator(mode="after")
    def _validate_bounds(self) -> "ContinuousSpace":
        if self.low >= self.high:
            raise ValueError(f"ContinuousSpace requires low < high; got low={self.low}, high={self.high}")
        return self
