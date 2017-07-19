"""
===========================
@Author  : aguajardo<aguajardo.me>
@Version: 1.0    24/03/2017
This is a module for RTDE Control
of a UR5 from Universal Robots.
===========================
"""

import logging
import os
import sys
import threading
import time

import Cense.Environment.Robot.rtde.rtde_config as rtde_config
import numpy as np

from Cense.Environment.Loop.loop import Loop
from Cense.Environment.Robot.rtde.rtde_py3 import RTDE


class IllegalPoseException(Exception):
    def __init__(self, *args):
        super(IllegalPoseException, self).__init__(*args)


class TerminalStateError(Exception):
    def __init__(self, *args):
        super(TerminalStateError, self).__init__(*args)

class SpawnedInTerminalStateError(TerminalStateError):
    def __init__(self, *args):
        super(SpawnedInTerminalStateError, self).__init__(*args)

class ExitedInTerminalStateError(TerminalStateError):
    def __init__(self, *args):
        super(ExitedInTerminalStateError, self).__init__(*args)

class RtdeController(object):
    # begin variable and object setup
    ROBOT_HOST = '137.226.189.172'
    ROBOT_PORT = 30004
    config_filename = fn = os.path.join(os.path.dirname(__file__), 'ur5_configuration.xml')

    RTDE_PROTOCOL_VERSION = 1

    ERROR_TRANSLATION = .001
    ERROR_ROTATION = 1 * np.pi / 180

    keep_running = True

    # CONSTRAINT_MIN = np.array([-.19, -.38, .17])
    # CONSTRAINT_MAX = np.array([.33, -.27, .7])

    CONSTRAINT_MIN = np.array([-.19, -.38, .07])
    CONSTRAINT_MAX = np.array([.33, -.27, .8])

    SPEED_FRACTION = .6

    connection = None

    target_pose = None

    # guarantees, that only one method will execute.
    # this is needed, because every method pauses synchronization to keep things from messing up
    lock = threading.Lock()

    def __init__(self, set_status_func):

        self.set_status_func = set_status_func
        self.set_status_func("Setup Robot")

        self.loop = Loop(set_status_func)

        conf = rtde_config.ConfigFile(self.config_filename)
        state_names, state_types = conf.get_recipe('state')

        state_names = [bytes(n, 'utf-8') for n in state_names]
        state_types = [bytes(t, 'utf-8') for t in state_types]

        target_pose_names, target_pose_types = conf.get_recipe('target_pose')

        target_pose_names = [bytes(n, 'utf-8') for n in target_pose_names]
        target_pose_types = [bytes(t, 'utf-8') for t in target_pose_types]

        abort_signal_names, abort_signal_types = conf.get_recipe('abort_signal')

        abort_signal_names = [bytes(n, 'utf-8') for n in abort_signal_names]
        abort_signal_types = [bytes(t, 'utf-8') for t in abort_signal_types]

        self.connection = RTDE(self.ROBOT_HOST, self.ROBOT_PORT)
        # end variable and object setup

        # Initiate connection
        self.connection.connect()

        # get_controller_version is used to know if minimum requirements are met
        self.connection.get_controller_version()

        # Compares protocol version of the robot with that of the program. Mismatch leads to system exit
        if not self.connection.negotiate_protocol_version(self.RTDE_PROTOCOL_VERSION):
            sys.exit()

        # Send configuration for output and input recipes
        self.connection.send_output_setup(state_names, state_types)
        self.target_pose = self.connection.send_input_setup(target_pose_names, target_pose_types)
        self.abort_signal = self.connection.send_input_setup(abort_signal_names, abort_signal_types)

        # Set input registers (double) to 0
        self.target_pose.input_double_register_0 = 0
        self.target_pose.input_double_register_1 = 0
        self.target_pose.input_double_register_2 = 0
        self.target_pose.input_double_register_3 = 0
        self.target_pose.input_double_register_4 = 0
        self.target_pose.input_double_register_5 = 0

        self.target_pose.speed_slider_mask = 1
        self.target_pose.speed_slider_fraction = 0.1

        self.target_pose.__dict__[b'speed_slider_mask'] = 1
        self.target_pose.__dict__[b'speed_slider_fraction'] = self.SPEED_FRACTION

        self.abort_signal.input_int_register_0 = 0

        self.connection.send_start()

        self.abort_signal.__dict__[b"input_int_register_0"] = 0
        self.connection.send(self.abort_signal)

    def current_pose(self):
        with self.lock:
            touching_wire = self.loop.is_touching_wire()

            logging.debug("Controller - current_pose")

            state = None

            while state is None:
                state = self.connection.receive()

            return np.array(state.__dict__[b'actual_TCP_pose']), touching_wire

    # moves tcp to specified pose
    # if wire is touched, move back to old position
    def move_to_pose(self, target_pose, force=False):
        with self.lock:

            if target_pose[0] < self.CONSTRAINT_MIN[0] or target_pose[0] > self.CONSTRAINT_MAX[0] \
                    or target_pose[1] < self.CONSTRAINT_MIN[1] or target_pose[1] > self.CONSTRAINT_MAX[1] \
                    or target_pose[2] < self.CONSTRAINT_MIN[2] or target_pose[2] > self.CONSTRAINT_MAX[2]:
                print("Illegal Pose!")
                raise IllegalPoseException

            state = None

            while state is None:
                state = self.connection.receive()

            start_pose = np.array(state.__dict__[b'actual_TCP_pose'])

            for i in range(6):
                self.target_pose.__dict__[b"input_double_register_" + str(i).encode()] = target_pose[i]

            if self.loop.is_touching_wire() and not force:
                raise SpawnedInTerminalStateError

            timestamp = time.time()

            while True:

                if self.loop.has_touched_wire(timestamp):
                    if not force:
                        break

                self.connection.send(self.target_pose)
                state = self.connection.receive()

                translation_deviation = np.sum(
                    np.absolute(state.__dict__[b'actual_TCP_pose'][:3] - target_pose[:3]))
                rotation_deviation = np.sum(np.absolute(((np.array(
                    state.__dict__[b'actual_TCP_pose'][3:] - target_pose[3:]) + np.pi) % (2*np.pi)) - np.pi))

                if translation_deviation < self.ERROR_TRANSLATION and rotation_deviation < self.ERROR_ROTATION:
                    break

            touched_wire = False

            if self.loop.has_touched_wire(timestamp) and not force:

                touched_wire = True

                self.abort_movement()

                for i in range(6):
                    self.target_pose.__dict__[b"input_double_register_" + str(i).encode()] = start_pose[i]

                while True:
                    self.connection.send(self.target_pose)
                    state = self.connection.receive()
                    translation_deviation = np.sum(
                        np.absolute(state.__dict__[b'actual_TCP_pose'][:3] - start_pose[:3]))
                    rotation_deviation = np.sum(np.absolute(((np.array(
                        state.__dict__[b'actual_TCP_pose'][3:] - start_pose[3:]) + np.pi) % (2*np.pi)) - np.pi))

                    if translation_deviation < self.ERROR_TRANSLATION \
                            and rotation_deviation < self.ERROR_ROTATION:
                        break

            # this function is supposed to ensure that the state is nonterminal
            if self.loop.is_touching_wire() and not force:
                raise ExitedInTerminalStateError

            return touched_wire

    def abort_movement(self):
        logging.debug('abort')

        self.abort_signal.__dict__[b"input_int_register_0"] = 1
        self.connection.send(self.abort_signal)

        while True:
            state = self.connection.receive()
            logging.debug(np.sum(np.absolute(state.__dict__[b'actual_TCP_speed'])))
            if np.sum(np.absolute(state.__dict__[b'actual_TCP_speed'])) < 0.01:
                break

        self.abort_signal.__dict__[b"input_int_register_0"] = 0
        self.connection.send(self.abort_signal)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)

    controller = RtdeController(print)

    START_POSE = np.array([.3, -.35, .458, 0, np.pi / 2, 0])
    GOAL_POSE = np.array([controller.CONSTRAINT_MIN[0] + .03, -.35, .458, 0, np.pi / 2, 0])
    # PREVIOUS_START_POSE = START_POSE

    while True:
        controller.move_to_pose(START_POSE)
        controller.move_to_pose(GOAL_POSE)



    # import matplotlib.pyplot as plt
    # from matplotlib.figure import Figure
    #
    # con = RtdeController(print)
    #
    # rotation = 30 * np.pi / 180
    # # rotation *= -1
    #
    # x = []
    # y = []
    #
    #
    # for i in range(1000):
    #     now = time.time()
    #     pose, _ = con.current_pose()
    #     pose[4] += rotation
    #     con.move_to_pose(pose)
    #
    #     execution_time = time.time()-now
    #
    #     rotation *= -1
    #
    #     x.append(i)
    #     y.append(execution_time)
    #
    #     plt.plot(x, y)
    #     plt.draw()
    #     plt.pause(0.001)
    #
    #     timeout = time.time() + 5
    #     while time.time() < timeout:
    #         pass


