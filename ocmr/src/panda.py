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


import mujoco
from src.robot_base import RobotBase


class Panda(RobotBase):
    def __init__(self, model, data):
        joint_names = [f'panda_joint{i + 1}' for i in range(7)]
        end_effector_name = 'panda_hand'
        super().__init__(model, data, 'panda', joint_names, end_effector_name)

    def set_end_effector_configuration(self, opening):
        self.data.joint('panda_finger_joint1').qpos = opening
        self.data.joint('panda_finger_joint2').qpos = opening
        mujoco.mj_forward(self.model, self.data)

    def close_gripper(self):
        self.set_end_effector_configuration(0.0)

    def open_gripper(self):
        self.set_end_effector_configuration(0.04)

    def set_gripper_controls(self, controls):
        self.data.actuator('pos_panda_finger_joint1').ctrl = controls[0]
        self.data.actuator('pos_panda_finger_joint2').ctrl = controls[1]

    @property
    def end_effector_configuration(self):
        finger_1 = self.data.joint('panda_finger_joint1').qpos[0].copy()
        finger_2 = self.data.joint('panda_finger_joint2').qpos[0].copy()
        return [finger_1, finger_2]
