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

import logging

from cloudai.core import BaseJob, System
from cloudai.util import CommandShell


class StandaloneSystem(System):
    """
    Class representing a Standalone system.

    This class is used for systems that execute commands directly without a job scheduler.
    """

    scheduler: str = "standalone"
    # Seconds between completion checks. Float rather than int so sub-second
    # values are expressible: trials that finish in tens of milliseconds are
    # badly served by the choice between 0 (spin without yielding, burning a
    # core that then competes with the trial) and 1 (25x longer than the work).
    monitor_interval: float = 1.0
    cmd_shell: CommandShell = CommandShell()

    def update(self) -> None:
        """
        Update the standalone system's state.

        This method is not typically used in standalone systems but is required for interface consistency.
        """
        pass

    def is_job_running(self, job: BaseJob) -> bool:
        """
        Check if a given standalone job is currently running.

        Polls the submitted process directly when the handle is available. The
        previous implementation shelled out to ``ps -p <pid>`` on every check,
        which the runner performs in a loop: measured at ~10 ms per check (two
        forks plus two pipe reads) against ~1 us for ``Popen.poll()``. On a
        workload of short trials that polling dominated the run -- profiling a
        100-trial scenario showed 811 process spawns, of which ~711 were these
        checks, accounting for roughly 42% of total wall clock.

        Polling the handle is also more correct. A pid identifies a process only
        while it lives, so with ``ps`` a recycled pid reads as "still running"
        and the runner waits forever; and the substring test below can match
        digits anywhere in ``ps`` output rather than the pid field.

        Args:
            job (BaseJob): The job to check.

        Returns:
            bool: True if the job is running, False otherwise.
        """
        process = getattr(job, "process", None)
        if process is not None:
            # poll() returns None while running, and reaps the child once it
            # exits, so this doubles as zombie cleanup.
            is_running = process.poll() is None
            logging.debug(f"Job {job.id} running status: {is_running} (via process handle)")
            return is_running

        # Jobs created without a handle -- dry-run mode, or a caller that only
        # recorded a pid -- still need an answer.
        command = f"ps -p {job.id}"
        logging.debug(f"Checking job status with command: {command}")
        stdout = self.cmd_shell.execute(command).communicate()[0]

        # Check if the job's PID is in the ps output
        is_running = str(job.id) in stdout
        logging.debug(f"Job {job.id} running status: {is_running}")

        return is_running

    def is_job_completed(self, job: BaseJob) -> bool:
        """
        Check if a given standalone job is completed.

        Args:
            job (BaseJob): The job to check.

        Returns:
            bool: True if the job is completed, False otherwise.
        """
        return not self.is_job_running(job)

    def kill(self, job: BaseJob) -> None:
        """
        Terminate a standalone job.

        Args:
            job (BaseJob): The job to be terminated.
        """
        process = getattr(job, "process", None)
        if process is not None:
            if process.poll() is None:
                logging.debug(f"Terminating job {job.id} via process handle")
                process.kill()
            # Always wait: reaps the child whether we just killed it or it had
            # already exited, so no zombie is left behind.
            process.wait()
            return

        cmd = f"kill -9 {job.id}"
        logging.debug(f"Executing termination command for job {job.id}: {cmd}")
        self.cmd_shell.execute(cmd)
