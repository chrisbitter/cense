from enum import Enum


#
# Defines an enumeration of allowed actions a robots tool center point can perform in a given world
#
class Action(Enum):
    # Moves the TCP 1 unit to the left
    LEFT = 0
    # Moves the TCP 1 unit to the right
    RIGHT = 1
    # Moves the TCP 1 unit upwards
    UP = 2
    # Moves the TCP 1 unit downwards
    DOWN = 3
    # Turns the TCP 45 degrees clockwise
    ROTATE_CLOCKWISE = 4
    # Turns the TCP 45 degrees counterclockwise
    ROTATE_COUNTER_CLOCKWISE = 5
