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
import argparse
import time
from threading import Thread
from mujoco_viewer import MujocoViewer
import sys
import pathlib
parent = pathlib.Path(__file__).parent.resolve()
sys.path.insert(1, str(parent))
sys.path.insert(1, str(parent / 'src'))
sys.path.insert(1, str(parent / 'src/optimisers'))
import utils
from src.optimisers.discovered_optimisers import all_optimisers
from src.panda import Panda
from src.results import ModelReductionResult


def reset_simulation():
    simulator.reset()
    mujoco_viewer.data = simulator.data
    mujoco_viewer.model = simulator.model


def infinitely_execute_trajectory_in_simulation(trajectory):
    while not keyboard_interrupted:
        reset_simulation()
        for arm_controls, gripper_controls in trajectory:
            while is_paused and not keyboard_interrupted:
                if keyboard_interrupted:
                    return
                continue
            simulator.execute_control(arm_controls, gripper_controls)
            time.sleep(simulator.timestep)
        time.sleep(1)

def visualise(trajectory):
    global keyboard_interrupted, mujoco_viewer
    keyboard_interrupted = False

    mujoco_viewer = MujocoViewer(simulator.model, simulator.data, width=700,
                                 height=500, title=f'Solving {args.model_filename}',
                                 hide_menus=True)

    main_thread = Thread(target=infinitely_execute_trajectory_in_simulation, args=(trajectory,))
    main_thread.start()

    global is_paused
    is_paused = False
    try:
        while mujoco_viewer.is_alive:
            is_paused = mujoco_viewer._paused
            mujoco_viewer.render()
            if args.show_names:
                utils.draw_object_names(mujoco_viewer, simulator)
    except KeyboardInterrupt:
        print('Quitting')

    keyboard_interrupted = True
    mujoco_viewer.close()
    main_thread.join()


def arg_parser():
    available_planners = ", ".join(list(all_optimisers.keys()))

    parser = argparse.ArgumentParser(description='Trajectory Optimiser Demo')
    parser.add_argument('model_filename', help='file name of the MuJoCo model to load.')
    parser.add_argument('optimiser_name', help=f'provide optimiser name (options: {available_planners}).')
    parser.add_argument('-g', '--group-num', type=int, default=5, help=f'number of objects in group')
    parser.add_argument('-s', '--save', action='store_true', help='save results and optimised trajectory to disk.')
    parser.add_argument('-v', '--view-solution', action='store_true', help='visualise the final trajectory.')
    parser.add_argument('-i', '--view-initial', action='store_true', help='visualise the initial trajectory.')
    parser.add_argument('-n', '--show-names', action='store_true', help='visualise the initial trajectory.')
    parser.add_argument('--debug', action='store_true', help='run in debug mode, printing useful information to screen.')

    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parser()

    keyboard_interrupted, mujoco_viewer = None, None

    logging_level = logging.DEBUG if args.debug else logging.ERROR
    logging.basicConfig(level=logging_level, format='%(message)s')

    # Create simulator from model_filename with a Robot class instance.
    simulator = utils.create_simulator(args.model_filename, Panda)

    # Factory for generating an optimiser from the optimiser_name using the
    # `simulator` object as the base simulator for the optimisation. Optimiser will
    # create copies of the `simulator` and will not change the state of that
    # specific object during optimisation.
    trajectory_optimiser = utils.optimiser_factory(args.optimiser_name, simulator)

    if args.group_num:
        trajectory_optimiser.group_size = args.group_num

    if args.view_initial:
        visualise(trajectory_optimiser.initial_trajectory)
    else:
        initial_trajectory = trajectory_optimiser.initial_trajectory
        optimisation_result = trajectory_optimiser.optimise(initial_trajectory)
        utils.print_optimisation_result(optimisation_result)

        if args.save:
            utils.save_data_to_file(args.model_filename, optimisation_result)
        if args.view_solution:
            visualise(optimisation_result.models[-1].best_trajectory)
