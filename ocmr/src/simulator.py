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


from copy import copy, deepcopy
import numpy as np
import time
import mujoco
from pathlib import Path
import os
from xml.etree import ElementTree
from typing import List
import utils


class Simulator:
    def __init__(self, model, model_file_name, Robot, delete_by_collisions=False):
        self.Robot = Robot
        self.model_filename = model_file_name
        self.model = model
        self.delete_by_collisions = delete_by_collisions
        self.data = None
        self.saved_data = None
        self.robot = None
        self.original_model_xml = self.read_xml()
        self.reduced_model_xml = self.read_xml()
        self.initialise_data()
        self.removed_objects = {}
        self.initialise_robot_configuration_from_config_file()

    def initialise_data(self):
        self.data = mujoco.MjData(self.model)
        self.saved_data = mujoco.MjData(self.model)
        self.robot = self.Robot(self.model, self.data)
        self.save_state()

    def initialise_robot_configuration_from_config_file(self):
        '''
        A YAML config file of the optimisation will dictate an initial
        configuration for the arm and the hand.  This method reads those
        configuraitons and sets the robot and the hand to the initial configs
        and saves the state.
        '''
        parameters = utils.get_optimisation_parameters()
        robot_initial_configuration = parameters['robot_initial_configuration']
        hand_initial_configuration = parameters['hand_initial_configuration']
        self.robot.set_arm_configuration(robot_initial_configuration)
        self.robot.set_end_effector_configuration(hand_initial_configuration)
        self.save_state()

    def get_object_forces(self, object_names, static_obstacle_names):
        # Find the IDs of all the objects of interest
        ids_of_objects_of_interest = set(
            (self.get_body_id(object_name) for object_name in object_names.union(static_obstacle_names)))

        forces = []
        number_of_contacts = self.data.ncon
        for i in range(number_of_contacts):
            contact = self.data.contact[i]
            body_in_contact_1 = self.model.body_rootid[self.model.geom_bodyid[contact.geom1]]
            body_in_contact_2 = self.model.body_rootid[self.model.geom_bodyid[contact.geom2]]

            # Avoid checking object with table/shelf bottom plate/supporting
            # surface, this will NOT ignore checks with the shelf boundary,
            # just with the shelf's bottom plate.
            contact_with_objects_of_interest = body_in_contact_1 in ids_of_objects_of_interest and body_in_contact_2 in ids_of_objects_of_interest

            if contact_with_objects_of_interest:
                body_force_torque = np.array([0.0] * 6)
                mujoco.mj_contactForce(self.model, self.data, i, body_force_torque)
                forces.append(body_force_torque)

        return deepcopy(forces)

    def get_object_names(self, prefix):
        all_names = set((name.decode('ascii') for name in self.model.names.split(b'\x00')))
        return set(filter(lambda name: name.startswith(prefix), all_names))

    def __copy__(self):
        simulator = Simulator(copy(self.model), self.model_filename, self.Robot)
        simulator.saved_data = copy(self.saved_data)
        simulator.reset()
        return simulator

    def reset(self):
        self.data = copy(self.saved_data)
        self.forward()
        self.robot.data = self.data

    def step(self, steps=1):
        mujoco.mj_step(self.model, self.data, nstep=steps)

    def forward(self):
        mujoco.mj_forward(self.model, self.data)

    def get_custom_text_data(self, custom_text_name):
        for i in range(self.model.ntext):
            adr = self.model.name_textadr[i]
            name = self.model.names[adr:adr + len(custom_text_name)].decode('ascii')
            if name == custom_text_name:
                adr = self.model.text_adr[i]
                size = self.model.text_size[i]
                value = self.model.text_data[adr:adr + size].decode('ascii')
                return value[:-1]

    def execute_in_realtime(self, trajectory):
        for arm_controls, gripper_controls in trajectory:
            self.execute_control(arm_controls, gripper_controls)
            time.sleep(self.timestep)

    def execute(self, trajectory):
        for arm_controls, gripper_controls in trajectory:
            self.execute_control(arm_controls, gripper_controls)

    def execute_control(self, arm_controls, gripper_controls):
        self.robot.set_arm_controls(deepcopy(arm_controls))
        self.robot.set_gripper_controls(deepcopy(gripper_controls))
        self.step()

    @property
    def timestep(self):
        return self.model.opt.timestep

    def get_body_id(self, object_name: str):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, object_name)

    def get_body_name(self, body_id: int):
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)

    def get_object_position(self, object_name: str):
        object_position = self.data.body(object_name).xpos.copy()
        return deepcopy(np.array(object_position))

    def get_object_orientation(self, object_name: str):
        object_orientation = self.data.body(object_name).xquat.copy()
        return deepcopy(np.array(object_orientation))

    def get_object_pose(self, object_name: str):
        object_position = deepcopy(self.get_object_position(object_name))
        object_orientation = deepcopy(self.get_object_orientation(object_name))
        return object_position, object_orientation

    def set_object_pose(self, object_name: str, x=None, y=None, z=None, quaternion=None):
        body_id = self.get_body_id(object_name)
        joint_index = self.model.body_jntadr[body_id]
        qpos_index = self.model.jnt_qposadr[joint_index]

        if x:
            self.data.qpos[qpos_index + 0] = x
        if y:
            self.data.qpos[qpos_index + 1] = y
        if z:
            self.data.qpos[qpos_index + 2] = z
        if quaternion:
            self.data.body(object_name).xquat[3] = quaternion[0]
            self.data.body(object_name).xquat[0] = quaternion[1]
            self.data.body(object_name).xquat[1] = quaternion[2]
            self.data.body(object_name).xquat[2] = quaternion[3]

        self.forward()

    def set_body_visual_transparency(self, body_name: str, transparency: float):
        body_id = self.get_body_id(body_name)
        number_of_geoms = self.model.body_geomnum[body_id]
        geom_starting_addr = self.model.body_geomadr[body_id]

        for _ in range(number_of_geoms):
            self.model.geom_rgba[geom_starting_addr][3] = transparency

    def _set_contact_attributes(self, body_name: str, contype: int, conaffinity: int):
        body_id = self.get_body_id(body_name)
        number_of_geoms = self.model.body_geomnum[body_id]
        geom_starting_addr = self.model.body_geomadr[body_id]

        for i in range(number_of_geoms):
            self.model.geom_contype[geom_starting_addr + i] = contype
            self.model.geom_conaffinity[geom_starting_addr + i] = conaffinity

    def get_contact_attributes(self, body_name: str):
        body_id = self.get_body_id(body_name)
        number_of_geoms = self.model.body_geomnum[body_id]
        geom_starting_addr = self.model.body_geomadr[body_id]

        result = {'contype': [], 'conaffinity': []}
        for i in range(number_of_geoms):
            result['contype'].append(self.model.geom_contype[geom_starting_addr + i].copy())
            result['conaffinity'].append(self.model.geom_conaffinity[geom_starting_addr + i].copy())

        return result

    def disable_all(self, object_names: List[str]):
        for object_name in object_names:
            self.disable(object_name, reload_model=False)

        if not self.delete_by_collisions:
            self.reload_model_from_model_reduced_xml()

    def disable(self, body_name: str, reload_model=True):
        if self.delete_by_collisions:
            self.disable_collisions(body_name)
            self.removed_objects[body_name] = True
        else:
            if body_name not in self.removed_objects:
                self.remove_object(body_name)

        if reload_model:
            self.reload_model_from_model_reduced_xml()

    def disable_collisions(self, body_name):
        self._set_contact_attributes(body_name, 4, 4)
        self.set_body_visual_transparency(body_name, 0.2)

    def enable_all(self, object_names):
        for object_name in object_names:
            self.enable(object_name, reload_model=False)

        if not self.delete_by_collisions:
            self.reload_model_from_model_reduced_xml()

    def enable(self, body_name, reload_model=True):
        if self.delete_by_collisions:
            self.enable_collisions(body_name)
            if body_name in self.removed_objects:
                self.removed_objects.pop(body_name)
        else:
            if body_name in self.removed_objects:
                self.add_object(body_name)

        if reload_model:
            self.reload_model_from_model_reduced_xml()

    def enable_collisions(self, body_name):
        self._set_contact_attributes(body_name, 1, 1)
        self.set_body_visual_transparency(body_name, 1.0)

    def in_collision(self, target, others):
        target_body_id = self.get_body_id(target)
        target_body_id = self.model.body_rootid[target_body_id]
        number_of_contacts = self.data.ncon

        for other in others:
            other_body_id = self.get_body_id(other)
            for i in range(number_of_contacts):
                contact = self.data.contact[i]
                body_in_contact_1 = self.model.body_rootid[self.model.geom_bodyid[contact.geom1]]
                body_in_contact_2 = self.model.body_rootid[self.model.geom_bodyid[contact.geom2]]

                if body_in_contact_1 == target_body_id and body_in_contact_2 == other_body_id:
                    return True
                elif body_in_contact_1 == other_body_id and body_in_contact_2 == target_body_id:
                    return True

        return False

    def save_state(self):
        """
        This method allows one to save the state of the simulator data, so
        next time simulator.reset() is called simulator is reset to that state.
        """
        self.saved_data = copy(self.data)

    def read_xml(self):
        root_path = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent.absolute()
        model_file_path = f'{root_path}/models/{self.model_filename}'
        return ElementTree.parse(model_file_path)

    def reload_model_from_xml(self, xml_string):
        arm_config = self.robot.arm_configuration
        hand_config = self.robot.end_effector_configuration

        models_path = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent.absolute() / 'models'
        os.chdir(models_path)

        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.initialise_data()
        self.robot.set_arm_configuration(arm_config)
        self.robot.set_end_effector_configuration(hand_config[0])
        self.save_state()

    def reload_model_from_model_reduced_xml(self):
        xml_string = ElementTree.tostring(self.reduced_model_xml.getroot(), encoding='unicode')
        self.reload_model_from_xml(xml_string)

    def reload_original_model(self):
        xml_string = ElementTree.tostring(self.original_model_xml.getroot(), encoding='unicode')
        self.reload_model_from_xml(xml_string)
        self.reduced_model_xml = deepcopy(self.original_model_xml)

    def remove_object(self, body_name):
        worldbody = self.reduced_model_xml.find('.//worldbody')
        body = worldbody.find(f'.//body[@name="{body_name}"]')
        self.removed_objects[body_name] = body
        worldbody.remove(body)

    def add_object(self, body_name):
        body = self.removed_objects[body_name]
        worldbody = self.reduced_model_xml.find('.//worldbody')
        worldbody.append(body)
        self.removed_objects.pop(body_name)
