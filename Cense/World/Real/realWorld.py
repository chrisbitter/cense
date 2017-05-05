from World.world import World
import RTDE_Controller_CENSE as rtde
import math
import numpy as np


class RealWorld(World):
    pi = math.pi
    scaling_constant = .01
    turn_constant = 45

    def move_left(self):
        current_pos = rtde.current_position()
        current_pos[0] -= RealWorld.scaling_constant
        rtde.move_to_position(current_pos)
        pass

    def move_right(self):
        current_pos = rtde.current_position()
        current_pos[0] += RealWorld.scaling_constant
        rtde.move_to_position(current_pos)
        pass

    def move_up(self):
        current_pos = rtde.current_position()
        current_pos[2] -= RealWorld.scaling_constant
        rtde.move_to_position(current_pos)
        pass

    def move_down(self):
        current_pos = rtde.current_position()
        current_pos[2] += RealWorld.scaling_constant
        rtde.move_to_position(current_pos)
        pass

    def turn_counter_clockwise(self):
        current_pos = rtde.current_position()
        current_pos[4] += RealWorld.pi*RealWorld.turn_constant/180
        rtde.move_to_position(current_pos)
        pass

    def turn_clockwise(self):
        current_pos = rtde.current_position()
        current_pos[4] -= RealWorld.pi*RealWorld.turn_constant/180
        rtde.move_to_position(current_pos)
        pass

    def get_state(self):
        # Ich weiss nicht was ich hier machen soll oder welche koordinaten sind in coordinates gespeichert.
        pass

    def go_to_coordinates(self):
        # Ich weiss nicht was ich hier machen soll oder welche koordinaten sind in coordinates gespeichert.
        pass

    def reset(self):
        rtde.go_start_via_path()
        pass

    def take_picture(self):
        rtde.disengage()
        rtde.go_camera()
        rtde.take_picture()
        rtde.go_start_disengaged()
        rtde.engage()
        pass


RealWorld = RealWorld()
print('CH1')
RealWorld.reset()


while True:
    # print('CH2')
    # RealWorld.move_up()
    # print('CH3')
    # RealWorld.move_down()
    # print('CH4')
    # RealWorld.move_right()
    # print('CH5')
    # RealWorld.move_left()
    # print('CH6')
    # RealWorld.turn_clockwise()
    # print('CH7')
    # RealWorld.turn_counter_clockwise()
    # print('CH8')
    RealWorld.take_picture()
