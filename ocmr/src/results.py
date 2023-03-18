# Optimisation with Chained Model Reduction (OCMR)
# Copyright (C) 2022 Rafael Papallas, University of Leeds
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 2 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, write to the Free Software Foundation, Inc., 51
# Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

from dataclasses import dataclass
from typing import List
from enum import Enum
from src.trajectory import Trajectory


class OptimisationOutcome(Enum):
    SUCCESS = 1
    FAILED_TIMEOUT = 2
    FAILED_LOCAL_MINIMA = 3
    INITIAL_IS_A_SOLUTION = 4
    FAILED_LOCAL_MINIMA_FULL_MODEL = 5


@dataclass
class ModelReductionResult:
    objects_in_model: list
    time: float


@dataclass
class OptimisationResult:
    iterations: int
    outcome: OptimisationOutcome
    best_trajectory: Trajectory
    rollout_times: list
    cost_history: list
    planning_time: float
    model_reduction_result: ModelReductionResult = None


@dataclass
class Result:
    optimiser_name: str
    world_file_name: str
    start_arm_configuration: list
    start_hand_configuration: list
    models: list
