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
import RTDE_Interface.rtde_client.rtde.rtde3_1 as rtde
import RTDE_Interface.rtde_client.rtde.rtde_config as rtde_config
from operator import sub, abs
import time
import _thread

# begin variable and object setup
ROBOT_HOST = '137.226.189.172'
ROBOT_PORT = 30004
config_filename = '../../Resources/ur5_configuration_CENSE_test.xml'

START_POSITION = [-0.387, -0.378, 0.560, 0, 0, 0]
DIS_START_POSITION = [-0.387, -0.328, 0.560, 0, 0, 0]
CENTER_POSITION = [0.11867, -0.30962, 0.71688, 0, 0, 0]
CAMERA_POSITION = [1.58721, -1.87299, 2.67201, -2.94720, -1.60205, 1.61322]
CAMERA_POSITION_POS = [0.11292, -0.26907, 0.19632, 0.99234, -0.01575, -0.03681]

RTDE_PROTOCOL_VERSION = 1

keep_running = True

MAX_ERROR = 0.0005

logging.getLogger().setLevel(logging.INFO)

conf = rtde_config.ConfigFile(config_filename)
state_names, state_types = conf.get_recipe('state')

state_names = [bytes(name, 'utf-8') for name in state_names]
state_types = [bytes(type, 'utf-8') for type in state_types]

setp_names, setp_types = conf.get_recipe('setp')

setp_names = [bytes(name, 'utf-8') for name in setp_names]
setp_types = [bytes(type, 'utf-8') for type in setp_types]

watchdog_names, watchdog_types = conf.get_recipe('watchdog')

watchdog_names = [bytes(name, 'utf-8') for name in watchdog_names]
watchdog_types = [bytes(type, 'utf-8') for type in watchdog_types]

joint_names, joint_types = conf.get_recipe('joint')

joint_names = [bytes(name, 'utf-8') for name in joint_names]
joint_types = [bytes(type, 'utf-8') for type in joint_types]

con = rtde.RTDE(ROBOT_HOST, ROBOT_PORT)
# end variable and object setup

# Initiate connection
con.connect()

# get_controller_version is used to know if minimum requirements are met
con.get_controller_version()

# Compares protocol version of the robot with that of the program. Mismatch leads to system exit
if not con.negotiate_protocol_version(RTDE_PROTOCOL_VERSION):
    sys.exit()

# Send configuration for output and input recipes
con.send_output_setup(state_names, state_types)
# print('check point 1')
setp = con.send_input_setup(setp_names, setp_types)
# print('check point 2')

# Send configuration for the watchdog timer (1 Hz) input recipe
watchdog = con.send_input_setup(watchdog_names, watchdog_types)

# Joint trigger: when 0 the points given are interpreted as pose, when 1 as joint angles
joint = con.send_input_setup(joint_names, joint_types)

# Set input registers (double) to 0
setp.input_double_register_0 = 0
setp.input_double_register_1 = 0
setp.input_double_register_2 = 0
setp.input_double_register_3 = 0
setp.input_double_register_4 = 0
setp.input_double_register_5 = 0

# Set input register for watchdog timer to 0 so that it can be reset
watchdog.input_int_register_0 = 0

# Set input register for joint to 0
joint.input_int_register_1 = 0


# Starts data sync
def start_sync():
    # start data exchange. If the exchange fails it returns 'Failed'
    if not con.send_start():
        return 'Failed'


# Pauses the data sync
def pause_sync():
    con.send_pause()


# Disconnects the RTDE
def disconnect_rtde():
    con.disconnect()


# current_position gives the current position of the TCP relative to the defined Cartesian plane in list format
def current_position():
    # Checks for the state of the connection
    state = con.receive()

    # If output config not initialized, RTDE synchronization is inactive, or RTDE is disconnected it returns 'Failed'
    if state is None:
        return 'Failed'
    # print(state.__dict__)
    # if the joint values are needed then return joint values
    if state.__dict__[b'output_int_register_1'] == 1:
        # If successful it returns the list with the current joint position
        return state.__dict__[b'actual_q']

    # If successful it returns the list with the current TCP position
    return state.__dict__[b'actual_TCP_pose']


# disengage moves the robot back away from the cable
def disengage():
    # Checks for the state of the connection
    state = con.receive()

    # If output config not initialized, RTDE synchronization is inactive, or RTDE is disconnected it returns 'Failed'
    if state is None:
        return 'Failed'

    # Set joint to 0
    joint.input_int_register_1 = 0
    con.send(joint)

    dis_pos = current_position()
    dis_pos[1] += 0.05
    move_to_position_no_append(dis_pos)

    return 'SUCCESS'


# engage() moves the robot forward onto the cable
def engage():
    # Checks for the state of the connection
    state = con.receive()

    # If output config not initialized, RTDE synchronization is inactive, or RTDE is disconnected it returns 'Failed'
    if state is None:
        return 'Failed'

    # Set joint to 0
    joint.input_int_register_1 = 0
    con.send(joint)

    dis_pos = current_position()
    dis_pos[1] -= 0.05
    move_to_position_no_append(dis_pos)

    return 'SUCCESS'


# This class hold the list of all positions
class Positions:
    start_sync()
    all_positions = [START_POSITION]
    state_idle = False
    if max(map(abs, map(sub, current_position(), CAMERA_POSITION_POS))) <= 0.005:
        at_camera = True
    else:
        at_camera = False
    print(at_camera)
    pause_sync()


# setp_to_list converts a serialized data object to a list
def setp_to_list(setp):
    list = []
    for i in range(0, 6):
        list.append(setp.__dict__["input_double_register_%i" % i])
    return list


# list_to_setp converts a list int0 serialized data object
def list_to_setp(setp, list):
    for i in range(0, 6):
        setp.__dict__["input_double_register_%i" % i] = list[i]
    return setp


# move_to_position changes the position and orientation of the TCP of the robot relative to the defined Cartesian plane
def move_to_position(new_pos):
    Positions.at_camera = False

    # Checks for the state of the connection
    state = con.receive()

    # If output config not initialized, RTDE synchronization is inactive, or RTDE is disconnected it returns 'Failed'
    if state is None:
        return 'Failed'

    # Set joint to 0
    joint.input_int_register_1 = 0
    con.send(joint)

    # Will try to move to position till current_position() is within a max error range from new_pos
    while max(map(abs, map(sub, current_position(), new_pos))) >= MAX_ERROR:
        # Checks for the state of the connection
        state = con.receive()

        # If output config not initialized, RTDE synchronization is inactive, or RTDE is disconnected it returns 'Failed'
        if state is None:
            return 'Failed'

        # The output_int_register_0 defines if the robot is in motion.
        if state.__dict__[b'output_int_register_0'] != 0:
            # Changes the value from setp to the new position
            list_to_setp(setp, new_pos)

            # Send new position
            con.send(setp)

        con.send(watchdog)

    # If successful the RTDE sync is paused, new position is added to all_positions, and it returns 'SUCCESS'
    Positions.all_positions.append(new_pos)
    return 'SUCCESS'


# move_to_position changes the position and orientation of the TCP of the robot relative to the defined Cartesian plane
def move_to_position_no_append(new_pos):
    Positions.at_camera = False

    # Will try to move to position till current_position() is within a max error range from new_pos
    while max(map(abs, map(sub, current_position(), new_pos))) >= MAX_ERROR:
        # print(current_position())
        # Checks for the state of the connection
        state = con.receive()

        # If output config not initialized, RTDE synchronization is inactive, or RTDE is disconnected it returns 'Failed'
        if state is None:
            return 'Failed'

        # The output_int_register_0 defines if the robot is in motion.
        if state.__dict__[b'output_int_register_0'] != 0:
            # Changes the value from setp to the new position
            list_to_setp(setp, new_pos)

            # Send new position
            con.send(setp)

        con.send(watchdog)

    # If successful the RTDE sync is paused, new position is added to all_positions, and it returns 'SUCCESS'
    return 'SUCCESS'


# go_start_via_path moves the robot back to the defined starting position through all recorded positions
def go_start_via_path():
    Positions.at_camera = False

    # Set joint to 0
    joint.input_int_register_1 = 0
    con.send(joint)

    # Makes a new list which is the reverses of all_positions to be able to go from end position to start position
    rev_all_positions = list(Positions.all_positions)
    rev_all_positions.reverse()
    move_to_position_no_append(rev_all_positions[0])

    # For all the positions in all_positions it implements the function move_to_position
    while len(rev_all_positions) != 1:
        move_response = 'Failed'

        # It will try the movement till 'SUCCESS' is returned
        while move_response == 'Failed':
            move_response = move_to_position_no_append(rev_all_positions[0])

        # It removes the position from all_positions
        rev_all_positions.remove(rev_all_positions[0])

    # Moves to start position
    move_to_position_no_append(rev_all_positions[0])

    # If successful all_positions will be cleared and redefined with initial position
    Positions.all_positions = list(rev_all_positions)
    return 'SUCCESS'


def go_start_disengaged():
    Positions.at_camera = False

    # Set joint to 0
    joint.input_int_register_1 = 0
    con.send(joint)

    # Moves to start position disengaged
    move_to_position_no_append(DIS_START_POSITION)

    return 'SUCCESS'
