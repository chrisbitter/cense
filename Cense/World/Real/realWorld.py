from Cense.World.world import World
from Cense.World.Camera.camera import Camera
import RTDE_Controller_CENSE as rtde
import math
import numpy as np


class RealWorld(World):
    Z_DECOUPLED = -5
    Z_COUPLED = 0

    START_POSITION = [-0.339, 0.387, Z_COUPLED, 0, 0, 0]
    GOAL_POSITION = [-0.339, 0.387, Z_COUPLED, 0, 0, 0]

    MAX_ERROR = .001
    MAX_ERROR_COUPLE = .01
    THRESHOLD_NEW_CHECKPOINT = .1
    THRESHOLD_OLD_CHECKPOINT = .005

    MAX_ERROR_SQUARED = MAX_ERROR * MAX_ERROR
    MAX_ERROR_COUPLE_SQUARED = MAX_ERROR_COUPLE * MAX_ERROR_COUPLE
    THRESHOLD_NEW_CHECKPOINT_SQUARED = THRESHOLD_NEW_CHECKPOINT * THRESHOLD_NEW_CHECKPOINT
    THRESHOLD_OLD_CHECKPOINT_SQUARED = THRESHOLD_OLD_CHECKPOINT * THRESHOLD_OLD_CHECKPOINT

    PUNISHMENT_WIRE = -1
    PUNISHMENT_DEAD_ZONE = -.1
    PUNISHMENT_OLD_CHECKPOINT = -.1
    REWARD_GOAL = 1
    REWARD_NEW_CHECKPOINT = .1
    REWARD_GENERIC = -.01

    STATE_DIMENSIONS = (50, 50)

    __checkpoints = []
    camera = None
    _state_terminal = True

    pi = math.pi
    move_constant = .01
    turn_constant = 45
    camera = None

    def __init__(self):
        self.camera = Camera()


    def move_left(self):
        current_pos = rtde.current_position()
        current_pos[0] -= RealWorld.move_constant
        rtde.move_to_position(current_pos)
        pass

    def move_right(self):
        current_pos = rtde.current_position()
        current_pos[0] += RealWorld.move_constant
        rtde.move_to_position(current_pos)
        pass

    def move_up(self):
        current_pos = rtde.current_position()
        current_pos[2] -= RealWorld.move_constant
        rtde.move_to_position(current_pos)
        pass

    def move_down(self):
        current_pos = rtde.current_position()
        current_pos[2] += RealWorld.move_constant
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

    def execute(self, action):
        # only move when state is not terminal
        if not self.in_terminal_state():

            if len(action) > 1:
                action = np.argmax(action)

            if action == 0:
                self.move_left()
            elif action == 1:
                self.move_right()
            elif action == 2:
                self.move_up()
            elif action == 3:
                self.move_down()
            elif action == 4:
                self.turn_clockwise()
            elif action == 5:
                self.turn_counter_clockwise()

            if self.is_touching_wire():
                reward = self.PUNISHMENT_WIRE

            elif self.is_in_dead_zone():
                reward = self.PUNISHMENT_DEAD_ZONE

            elif self.is_at_old_checkpoint():
                reward = self.PUNISHMENT_OLD_CHECKPOINT
                self.regress_checkpoints()

            elif self.is_at_goal():
                reward = self.REWARD_GOAL

            elif self.is_at_new_checkpoint():
                reward = self.REWARD_NEW_CHECKPOINT
                self.advance_checkpoints()
            else:
                reward = self.REWARD_GENERIC
        else:
            reward = 0

        state, terminal = self.observe()

        return state, reward, terminal

    def observe(self):
        return self.camera.capture_image(), self.in_terminal_state()

    def in_terminal_state(self):
            return self.is_touching_wire() | self.is_in_dead_zone() | \
                    self.is_at_goal()

    def is_touching_wire(self):
        # todo: get wire/loop circuit signal
        wire_signal = True

        if wire_signal:
            self.logging.warning("Loop is touching the wire!")
        raise NotImplementedError

    def is_at_goal(self):
        raise NotImplementedError

    def is_in_dead_zone(self):
        raise NotImplementedError

    def invert_game(self):
        self.logging.debug("invert_game")

        rtde.disengage()
        rtde.rotate_180()
        self.engage()

        # switch start and goal
        temp = self.START_POSITION
        self.START_POSITION = self.GOAL_POSITION
        self.GOAL_POSITION = temp

        # reset checkpoints
        self.__checkpoints = [self.START_POSITION]


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
