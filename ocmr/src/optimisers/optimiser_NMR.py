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

from src.optimisers.discovered_optimisers import register_optimiser
from src.trajectory_optimiser_base import TrajectoryOptimiserBase
from src.results import Result


@register_optimiser
class NMR(TrajectoryOptimiserBase):
    def optimise(self, trajectory):
        start_arm_configuration = self.simulators['main'].robot.arm_configuration
        start_hand_configuration = self.simulators['main'].robot.end_effector_configuration

        result = super().optimise(trajectory)

        return Result(self.__class__.__name__, 
                      self.simulators['main'].model_filename,
                      start_arm_configuration, 
                      start_hand_configuration,
                      [result])


