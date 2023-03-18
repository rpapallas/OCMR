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


import logging
import time
import numpy as np
from copy import deepcopy
from src.optimisers.discovered_optimisers import register_optimiser
from src.results import Result, OptimisationOutcome, ModelReductionResult
from src.trajectory_optimiser_base import TrajectoryOptimiserBase


class OCMR(TrajectoryOptimiserBase):
    def remove_all_objects_from_all_rollout_simulators(self):
        for i in range(self.num_of_rollouts):
            sim_name = f'rollout_{i}'
            self.disable_all_objects_except_goal_object(sim_name)

    def add_objects_to_all_rollout_simulators(self, object_names):
        for i in range(self.num_of_rollouts):
            sim_name = f'rollout_{i}'
            self.simulators[sim_name].enable_all(object_names)

    def optimise(self, trajectory):
        start_arm_configuration = self.simulators['full_model'].robot.arm_configuration
        start_hand_configuration = self.simulators['full_model'].robot.end_effector_configuration

        remaining_objects = list(self.all_obstacle_names)
        total_model_reduction_time = 0.0
        original_distance_to_goal = self.optimisation_parameters['distance_to_goal']
        original_time_limit = self.optimisation_parameters['time_limit']
        total_planning_time = 0.0
        optimisation_results = []

        # If initial traj is a solution, just return the result of planning with full model.
        full_model_previous_rollout, _ = self.rollout(trajectory, with_simulator_name='full_model')
        if full_model_previous_rollout.distance_to_goal <= original_distance_to_goal:
            result = super().optimise(trajectory)
            return Result(self.__class__.__name__, 
                          self.simulators['main'].model_filename,
                          start_arm_configuration, 
                          start_hand_configuration,
                          [result])

        # Otherwise, start model reduction and planning.
        self.remove_all_objects_from_all_rollout_simulators()

        while len(remaining_objects) > 0:
            if len(remaining_objects) <= self.group_size:
                objects_to_add = remaining_objects[:self.group_size]
                model_reduction_result = ModelReductionResult(objects_to_add, 0.0)
            else:
                model_reduction_result = self.get_next_group_to_add(remaining_objects, trajectory)
                total_model_reduction_time += model_reduction_result.time
                objects_to_add = model_reduction_result.objects_in_model

            self.add_objects_to_all_rollout_simulators(objects_to_add)
            logging.info(f'Object added: {",".join(objects_to_add)!r}')

            for object_name in objects_to_add:
                remaining_objects.remove(object_name)

            self.optimisation_parameters['time_limit'] = 20
            if len(remaining_objects) == 0:
                self.optimisation_parameters['time_limit'] = original_time_limit - total_planning_time

            optimisation_result = super().optimise(trajectory)

            start_time = time.time()
            full_model_rollout, _ = self.rollout(optimisation_result.best_trajectory, 'full_model')
            end_time = time.time()

            optimisation_result.planning_time += end_time - start_time
            total_planning_time += optimisation_result.planning_time

            is_going_downhill = full_model_rollout.cost <= full_model_previous_rollout.cost
            logging.info(f'Traj from reduced model gets us downhill in full model: {is_going_downhill}, cost diff: {full_model_rollout.cost - full_model_previous_rollout.cost}, distance to goal: {full_model_rollout.distance_to_goal:.2f}')
            optimisation_result.model_reduction_result = model_reduction_result
            optimisation_results.append(optimisation_result)
            if is_going_downhill:
                trajectory = deepcopy(optimisation_result.best_trajectory)
                full_model_previous_rollout = full_model_rollout
            if full_model_rollout.distance_to_goal <= original_distance_to_goal:
                optimisation_result.outcome = OptimisationOutcome.SUCCESS
                break

        return Result(f'{self.__class__.__name__}-G{self.group_size}', self.simulators['full_model'].model_filename,
                      start_arm_configuration, start_hand_configuration,
                      optimisation_results)


class OCMR_PositionBased(OCMR):
    def get_next_group_to_add(self, objects_to_rank, traj):
        start = time.time()
        object_distances = self.compute_object_distances(objects_to_rank, traj)
        sorted_object_names = [(name, np.mean(dist)) for name, dist in object_distances.items()]
        sorted_object_names.sort(key=lambda name_distance: name_distance[1])
        object_names_ranked = [name_distance[0] for name_distance in sorted_object_names]
        end = time.time()
        return ModelReductionResult(object_names_ranked[:self.group_size], end - start)

# Cost-based
# Mean distance (goal object + hand) during traj execution
class Grouping:
    def compute_object_distances(self, object_names, traj):
        distances = {
            object_name: []
            for object_name
            in object_names
        }

        self.simulators['main'].reset()

        for arm_controls, gripper_controls in traj:
            goal_object_position = self.simulators['main'].get_object_position(self.goal_object_name)
            hand_position = self.simulators['main'].robot.end_effector_position

            for object_name in object_names:
                other_object_position = self.simulators['main'].get_object_position(object_name)
                distance_to_hand = np.linalg.norm(hand_position - other_object_position)
                distance_to_goal_object = np.linalg.norm(goal_object_position - other_object_position)
                mean_distance = (distance_to_hand + distance_to_goal_object) / 2
                distances[object_name].append(mean_distance)

            self.simulators['main'].execute_control(arm_controls, gripper_controls)

        self.simulators['main'].reset()
        return distances


@register_optimiser
class OCMR2(OCMR_PositionBased, Grouping):
    def __init__(self, *args):
        self.group_size = 5
        super().__init__(*args)
