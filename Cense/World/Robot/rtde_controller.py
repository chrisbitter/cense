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

class IllegalPoseException(Exception):
    pass


class RTDE_Controller:
    # begin variable and object setup
    ROBOT_HOST = '137.226.189.172'
    ROBOT_PORT = 30004
    config_filename = fn = os.path.join(os.path.dirname(__file__), 'ur5_configuration_CENSE_test.xml')

    RTDE_PROTOCOL_VERSION = 1

    ERROR = .001

    keep_running = True

    CONSTRAINT_MIN = np.array([-.23, -.26, 0.35])
    CONSTRAINT_MAX = np.array([.36, -.19, 0.85])

    connection = None

    setp = None

    def __init__(self):
        print("Connecting Robot")
        logging.getLogger().setLevel(logging.ERROR)

        conf = rtde_config.ConfigFile(self.config_filename)
        state_names, state_types = conf.get_recipe('state')

        state_names = [bytes(name, 'utf-8') for name in state_names]
        state_types = [bytes(type, 'utf-8') for type in state_types]

        setp_names, setp_types = conf.get_recipe('setp')

        setp_names = [bytes(name, 'utf-8') for name in setp_names]
        setp_types = [bytes(type, 'utf-8') for type in setp_types]

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

        self.setp = self.connection.send_input_setup(setp_names, setp_types)

        # Set input registers (double) to 0
        self.setp.input_double_register_0 = 0
        self.setp.input_double_register_1 = 0
        self.setp.input_double_register_2 = 0
        self.setp.input_double_register_3 = 0
        self.setp.input_double_register_4 = 0
        self.setp.input_double_register_5 = 0

        self.connection.send_start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Disconnecting Robot")
        self.connection.disconnect()

    # Starts data sync
    def start_sync(self):
        # start data exchange. If the exchange fails it returns 'Failed'
        if not self.connection.send_start():
            return False

    # Pauses the data sync
    def pause_sync(self):
        self.connection.send_pause()

    # Disconnects the RTDE
    def disconnect_rtde(self):
        self.connection.disconnect()

    # current_position gives the current position of the TCP relative to the defined Cartesian plane in list format
    def current_pose(self):
        # Checks for the state of the connection
        # wait until robot finishes move
        while True:
            state = self.connection.receive()
            if state.__dict__[b'output_int_register_0'] == 0:
                break

        # If output config not initialized, RTDE synchronization is inactive, or RTDE is disconnected it returns 'Failed'
        if state is None:
            return None

        # If successful it returns the list with the current TCP position
        return np.array(state.__dict__[b'actual_TCP_pose'])

    # move_to_position changes the position and orientation of the TCP of the robot relative to the defined Cartesian plane
    def move_to_pose(self, new_pose):
        #print("Move: ", new_pose)

        if new_pose[0] < self.CONSTRAINT_MIN[0] or new_pose[0] > self.CONSTRAINT_MAX[0] \
                or new_pose[1] < self.CONSTRAINT_MIN[1] or new_pose[1] > self.CONSTRAINT_MAX[1] \
                or new_pose[2] < self.CONSTRAINT_MIN[2] or new_pose[2] > self.CONSTRAINT_MAX[2]:
            # print(new_pose)
            # print(new_pose[0] < self.CONSTRAINT_MIN[0], new_pose[0] > self.CONSTRAINT_MAX[0],
            #       new_pose[1] < self.CONSTRAINT_MIN[1], new_pose[1] > self.CONSTRAINT_MAX[1],
            #       new_pose[2] < self.CONSTRAINT_MIN[2], new_pose[2] > self.CONSTRAINT_MAX[2])
            raise IllegalPoseException

        # Checks for the state of the connection
        state = self.connection.receive()

        # If output config not initialized, RTDE synchronization is inactive, or RTDE is disconnected it throws an connection error
        if state is None:
            raise ConnectionError

        # set registers to new pose values
        for i in range(6):
            self.setp.__dict__[b"input_double_register_" + str(i).encode()] = new_pose[i]

        #while np.linalg.norm(self.current_pose() - new_pose) > self.ERROR:

        # Send new position
        self.connection.send(self.setp)

        # wait until robot finishes move
        while True:
            state = self.connection.receive()
            if state.__dict__[b'output_int_register_0'] == 0:
                break
