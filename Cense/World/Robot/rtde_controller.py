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


class IllegalPoseException(Exception):
    pass


class RTDE_Controller:
    # begin variable and object setup
    ROBOT_HOST = '137.226.189.172'
    ROBOT_PORT = 30004
    config_filename = 'ur5_configuration_CENSE_test.xml'

    RTDE_PROTOCOL_VERSION = 1

    keep_running = True

    X_MIN = -.147
    X_MAX = .287

    Y_MIN = -.333
    Y_MAX = -.284

    Z_MIN = 0.492
    Z_MAX = 0.873

    connection = None

    setp = None

    def __init__(self):
        print("Connecting Robot")
        logging.getLogger().setLevel(logging.INFO)

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
        state = self.connection.receive()

        # If output config not initialized, RTDE synchronization is inactive, or RTDE is disconnected it returns 'Failed'
        if state is None:
            return None

        # If successful it returns the list with the current TCP position
        return state.__dict__[b'actual_TCP_pose']

    # setp_to_list converts a serialized data object to a list
    def setp_to_list(self, setp):
        list = []
        for i in range(0, 6):
            list.append(setp.__dict__[b"input_double_register_%i" % i])
        return list

    # list_to_setp converts a list int0 serialized data object
    def list_to_setp(self, setp, list):
        for i in range(6):
            setp.__dict__[b"input_double_register_%i" % i] = list[i]
        return setp

    # move_to_position changes the position and orientation of the TCP of the robot relative to the defined Cartesian plane
    def move_to_pose(self, new_pose):

        if new_pose[0] < self.X_MIN or new_pose[0] > self.X_MAX \
                or new_pose[1] < self.Y_MIN or new_pose[1] > self.Y_MAX \
                or new_pose[2] < self.Z_MIN or new_pose[2] > self.Z_MAX:
            print(new_pose[0] < self.X_MIN, new_pose[0] > self.X_MAX,
                  new_pose[1] < self.Y_MIN, new_pose[1] > self.Y_MAX,
                  new_pose[2] < self.Z_MIN, new_pose[2] > self.Z_MAX)
            raise IllegalPoseException

        # Checks for the state of the connection
        state = self.connection.receive()

        # If output config not initialized, RTDE synchronization is inactive, or RTDE is disconnected it throws an connection error
        if state is None:
            raise ConnectionError

        self.list_to_setp(self.setp, new_pose)

        # Send new position
        self.connection.send(self.setp)

        # wait until robot finishes move
        while True:
            state = self.connection.receive()
            if state.__dict__[b'output_int_register_0'] == 0:
                break
