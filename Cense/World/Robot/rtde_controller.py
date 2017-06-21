"""
===========================
@Author  : aguajardo<aguajardo.me>
@Version: 1.0    24/03/2017
This is a module for RTDE Control
of a UR5 from Universal Robots.
===========================
"""

import sys
import logging
from Cense.World.Robot.rtde_client.rtde.rtde3_1 import RTDE
import Cense.World.Robot.rtde_client.rtde.rtde_config as rtde_config
from operator import sub, abs
import time
import numpy as np
import os
import threading
from Cense.World.Loop.loop import Loop

class IllegalPoseException(Exception):
    def __init__(self, *args):
        super().__init__(self, *args)

class TerminalStateError(Exception):
    def __init__(self, *args):
        super().__init__(self, *args)

class RTDE_Controller:
    # begin variable and object setup
    ROBOT_HOST = '137.226.189.172'
    ROBOT_PORT = 30004
    config_filename = fn = os.path.join(os.path.dirname(__file__), 'ur5_configuration_CENSE_test.xml')

    RTDE_PROTOCOL_VERSION = 1

    ERROR = .001

    keep_running = True

    CONSTRAINT_MIN = np.array([-.24, -.38, .2])
    CONSTRAINT_MAX = np.array([.33, -.27, .68])

    connection = None

    target_pose = None

    # guarantees, that only one method will execute.
    # this is needed, because every method pauses synchronization to keep things from messing up
    lock = threading.Lock()

    def __init__(self):
        print("Connecting Robot")

        self.loop = Loop()

        conf = rtde_config.ConfigFile(self.config_filename)
        state_names, state_types = conf.get_recipe('state')

        state_names = [bytes(name, 'utf-8') for name in state_names]
        state_types = [bytes(type, 'utf-8') for type in state_types]

        target_pose_names, target_pose_types = conf.get_recipe('target_pose')

        target_pose_names = [bytes(name, 'utf-8') for name in target_pose_names]
        target_pose_types = [bytes(type, 'utf-8') for type in target_pose_types]

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

        # Set input registers (double) to 0
        self.target_pose.input_double_register_0 = 0
        self.target_pose.input_double_register_1 = 0
        self.target_pose.input_double_register_2 = 0
        self.target_pose.input_double_register_3 = 0
        self.target_pose.input_double_register_4 = 0
        self.target_pose.input_double_register_5 = 0

        self.connection.send_start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Disconnecting Robot")
        self.connection.disconnect()

    def is_moving(self):
        with self.lock:
            logging.debug("Controller - is_moving")

            state = self.connection.receive()

            return state.__dict__[b'output_int_register_0'] == 1

    # current_position gives the current position of the TCP relative to the defined Cartesian plane in list format
    def current_pose(self):
        with self.lock:
            touching_wire = self.loop.is_touching_wire()

            logging.debug("Controller - current_pose")

            # Checks for the state of the connection
            # wait until robot finishes move
            while True:
                logging.debug("waiting for state")
                state = self.connection.receive()
                if state.__dict__[b'output_int_register_0'] == 0:
                    break

            # If output config not initialized, RTDE synchronization is inactive, or RTDE is disconnected it returns 'Failed'
            if state is None:
                return None, touching_wire

            # If successful it returns the list with the current TCP position
            return np.array(state.__dict__[b'actual_TCP_pose']), touching_wire

    # move_to_position changes the position and orientation of the TCP of the robot relative to the defined Cartesian plane
    # if wire is touched, move back to old position
    def move_to_pose(self, target_pose, steps=10, force=False):

        with self.lock:

            if target_pose[0] < self.CONSTRAINT_MIN[0] or target_pose[0] > self.CONSTRAINT_MAX[0] \
                    or target_pose[1] < self.CONSTRAINT_MIN[1] or target_pose[1] > self.CONSTRAINT_MAX[1] \
                    or target_pose[2] < self.CONSTRAINT_MIN[2] or target_pose[2] > self.CONSTRAINT_MAX[2]:
                raise IllegalPoseException

            # Checks for the state of the connection
            state = self.connection.receive()

            # If output config not initialized, RTDE synchronization is inactive, or RTDE is disconnected it throws an connection error
            if state is None:
                raise ConnectionError

            start_pose = np.array(state.__dict__[b'actual_TCP_pose'])
            step = (np.array(target_pose) - start_pose) / steps

            # if robot is touching wire, its stuck!
            if self.loop.is_touching_wire() and not force:
                raise TerminalStateError

            touched_wire = False

            for s in range(steps):
                if not self.loop.has_touched_wire() or force:
                    new_pose = start_pose + (s+1)*step
                    # set registers to new pose values
                    for i in range(6):
                        self.target_pose.__dict__[b"input_double_register_" + str(i).encode()] = new_pose[i]
                        # Send new position
                        self.connection.send(self.target_pose)
                else:
                    # if robot touched wire, move back to old pose
                    touched_wire = True
                    for i in range(6):
                        self.target_pose.__dict__[b"input_double_register_" + str(i).encode()] = start_pose[i]

                    self.connection.send(self.target_pose)

            # if robot ended up in a terminal state and wasn't forced there, something went wrong
            if self.loop.is_touching_wire() and not force:
                raise TerminalStateError

            return touched_wire

if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)

    con = RTDE_Controller()

    timeout = 10
    rotation = .52

    for _ in range(100):
        #print(timeout)
        pose = con.current_pose()
        pose[4] += rotation
        con.move_to_pose(pose)

        rotation *= -1

        now = time.time()
        while time.time() - now < timeout:
            pass
