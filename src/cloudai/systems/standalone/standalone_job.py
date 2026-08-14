# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import subprocess
from dataclasses import dataclass, field
from typing import Optional

from cloudai.core import BaseJob


@dataclass
class StandaloneJob(BaseJob):
    """A job class for standalone execution."""

    # The live handle to the submitted process. Without it, completion has to be
    # checked by shelling out to ``ps -p <pid>``, which costs two forks and two
    # pipe reads (~10 ms) where ``Popen.poll()`` is a single waitpid (~1 us).
    # Retaining the handle also removes a correctness hazard: a pid is only
    # unique while its process lives, so a recycled pid can make a finished job
    # look like it is still running.
    #
    # Excluded from equality and repr: two jobs are the same job by test_run and
    # id, and a Popen's repr is noise in logs.
    process: Optional[subprocess.Popen] = field(default=None, compare=False, repr=False)
