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

class RTDE_Controller:
    # begin variable and object setup
    ROBOT_HOST = '169.254.203.187'
    ROBOT_PORT = 30004
    config_filename = 'ur5_configuration_CENSE_test.xml'

    START_POSITION = [-0.387, -0.378, 0.560, 0, 0, 0]
    DIS_START_POSITION = [-0.387, -0.328, 0.560, 0, 0, 0]
    CENTER_POSITION = [0.11867, -0.30962, 0.71688, 0, 0, 0]
    CAMERA_POSITION = [1.58721, -1.87299, 2.67201, -2.94720, -1.60205, 1.61322]
    CAMERA_POSITION_POS = [0.11292, -0.26907, 0.19632, 0.99234, -0.01575, -0.03681]

    RTDE_PROTOCOL_VERSION = 1

    keep_running = True

    MAX_ERROR = 0.0005

    connection = None

    setp = None

    def __init__(self):
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

        print("send input setup")
        self.setp = self.connection.send_input_setup(setp_names, setp_types)
        print("done")

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
    def current_position(self):

        print("Current pose")
        # Checks for the state of the connection
        state = self.connection.receive()

        print("state: ", state)

        # If output config not initialized, RTDE synchronization is inactive, or RTDE is disconnected it returns 'Failed'
        if state is None:
            return None
        # print(state.__dict__)

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
        # Checks for the state of the connection
        state = self.connection.receive()

        # If output config not initialized, RTDE synchronization is inactive, or RTDE is disconnected it returns 'Failed'
        if state is None:
            return 'Failed'

        self.list_to_setp(self.setp, new_pose)

        # Send new position
        self.connection.send(self.setp)

        while True:

            state = self.connection.receive()
            if state.__dict__[b'output_int_register_0'] == 0:
                break

        # # Will try to move to position till current_position() is within a max error range from new_pos
        # while max(map(abs, map(sub, self.current_position(), new_pose))) >= self.MAX_ERROR:
        #     # Checks for the state of the connection
        #     state = self.connection.receive()
        #
        #     # If output config not initialized, RTDE synchronization is inactive, or RTDE is disconnected it returns 'Failed'
        #     if state is None:
        #         return 'Failed'
        #
        #     # The output_int_register_0 defines if the robot is in motion.
        #     if state.__dict__[b'output_int_register_0'] != 0:
        #         # Changes the value from setp to the new position
        #         self.list_to_setp(self.setp, new_pose)
        #
        #         # Send new position
        #         self.connection.send(self.setp)
        #
        #         # self.connection.send(watchdog)

        # If successful the RTDE sync is paused, new position is added to all_positions, and it returns 'SUCCESS'
        return 'SUCCESS'
