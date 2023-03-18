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
from typing import Dict
from pyquaternion import Quaternion


@dataclass
class ObjectState:
    position: list
    orientation: list


@dataclass
class State:
    robot_configuration: List
    robot_joint_velocities: List
    hand_position: List
    hand_orientation: Quaternion
    objects: Dict
    object_forces: List
    robot_in_collision_with_static_obstacles: bool


class StateSequence(list):
    pass
