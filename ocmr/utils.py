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


import csv
import sys
from pathlib import Path
import os
import mujoco
import numpy
import yaml
from src.optimisers.discovered_optimisers import all_optimisers
from src.simulator import Simulator


def draw_object_names(viewer, simulator, groups=None):
    mjtype = mujoco.mjtGeom.mjGEOM_LABEL
    if groups:
        for index, group in enumerate(groups):
            for object_name in group.names:
                position, _ = simulator.get_object_pose(object_name)
                position[1] += 0.01
                viewer.add_marker(pos=position, type=mjtype, label=f'{object_name.split("_")[-1]} G{index + 1}')
    else:
        for object_name in simulator.get_object_names(prefix="movable_obstacle_"):
            position, _ = simulator.get_object_pose(object_name)
            viewer.add_marker(pos=position, type=mjtype, label=object_name.split("_")[-1])


def optimiser_factory(optimiser_name, simulator):
    if optimiser_name.upper() in all_optimisers.keys():
        return all_optimisers[optimiser_name.upper()](simulator)
    else:
        available_planners = ", ".join(list(all_optimisers.keys()))
        sys.exit(f'Unrecognised optimiser name. Possible planners: {available_planners}')


def create_simulator(model_filename, Robot):
    model = load_model(model_filename)
    return Simulator(model, model_filename, Robot)


def load_model(model_filename):
    root_path = Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute()
    return mujoco.MjModel.from_xml_path(f'{root_path}/models/{model_filename}')


def get_optimisation_parameters():
    root_path = Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute()
    with open(f'{root_path}/config/shelf.yaml') as file:
        optimiser_parameters = yaml.safe_load(file)
        return optimiser_parameters


def print_optimisation_result(result):
    total_number_of_iterations = 0
    total_rollout_times = []

    indent = "    "
    global_planning_time = 0.0
    for i, optimisation_result in enumerate(result.models):
        print(f'Model {i+1}')
        print(f'{indent}Outcome: {optimisation_result.outcome.name}')

        total_number_of_iterations += optimisation_result.iterations
        total_rollout_times.append(sum(optimisation_result.rollout_times))

        average_rollout_time = numpy.mean(optimisation_result.rollout_times)
        print(f'{indent}Average rollout time: {average_rollout_time:.3f} (in {optimisation_result.iterations} iterations)')

        print(f'{indent}Planning time: {optimisation_result.planning_time:.3f}')
        global_planning_time += optimisation_result.planning_time
        model_reduction_result = optimisation_result.model_reduction_result
        if model_reduction_result:
            time_elapsed = model_reduction_result.time
            print(f'{indent}Model reduction time: {time_elapsed:.3f}.')

    global_average_rollout_time = sum(total_rollout_times) / total_number_of_iterations if total_number_of_iterations > 0 else 0.0
    print(f'Global average rollout time: {global_average_rollout_time:.3f} (in {total_number_of_iterations} iterations)')
    print(f'Global planning time: {global_planning_time:.3f}')


def get_next_experiment_id():
    root_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute()
    path = root_dir / 'results' / 'next_experiment_id.txt'
    experiment_id = int(path.read_text())
    return experiment_id


def update_next_experiment_id_file():
    root_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute()
    path = root_dir / 'results' / 'next_experiment_id.txt'
    experiment_id = int(path.read_text())
    path.write_text(str(experiment_id + 1))


def save_data_to_file(world_name, result):
    root_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent.absolute()

    experiment_id = get_next_experiment_id()

    path = root_dir / 'results' / 'results.csv'
    with open(path, 'a') as results_file:
        csv_writer = csv.writer(results_file, delimiter=',', quoting=csv.QUOTE_ALL)
        csv_writer.writerow([experiment_id,
                             world_name,
                             result.optimiser_name,
                             result.models[-1].outcome.name,
                             [opt_result.rollout_times for opt_result in result.models],
                             sum([opt_result.planning_time for opt_result in result.models]),
                             sum([opt_result.model_reduction_result.time for opt_result in result.models if opt_result.model_reduction_result]),
                             [opt_result.iterations for opt_result in result.models],
                             ])

        update_next_experiment_id_file()
