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
from src.cost_sequence import CostSequence
from src.trajectory import Trajectory


@dataclass
class RollOut:
    trajectory: Trajectory
    cost_sequence: CostSequence = None
    distance_to_goal: float = 0.0

    @property
    def cost(self):
        return self.cost_sequence.total
