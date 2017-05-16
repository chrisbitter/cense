# from Cense.World.world import World
from Cense.World.Camera.camera import Camera
# import RTDE_Controller_CENSE as rtde
import math
import numpy as np
import logging


class TerminalStateError(Exception):
    def __init__(self, *args):
        super().__init__(self, *args)


class DummyWorld(object):
    Z_DISENGAGED = -5
    Z_ENGAGED = 0

    # todo: Find correct coordinates
    START_POSE = [-0.339, 0.387, Z_ENGAGED, 0, 0, 0]
    GOAL_POSE = [-0.339, 0.387, Z_ENGAGED, 0, 0, 0]

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

    STATE_DIMENSIONS = (50, 50, )
    ACTIONS = 6

    __checkpoints = []
    camera = None

    debug_enter_nonterminal = False

    translation_constant = .01
    rotation_constant = 45

    def __init__(self):
        self.camera = Camera()

    def init_nonterminal_state(self):
        self.debug_nonterminal = True

    def execute(self, action):

        current_pos = np.zeros(6)

        if action == 0:
            # Left
            current_pos[0] -= self.translation_constant
        elif action == 1:
            # Right
            current_pos[0] += self.translation_constant
        elif action == 2:
            # Up
            current_pos[2] -= self.translation_constant
        elif action == 3:
            # Down
            current_pos[2] += self.translation_constant
        elif action == 4:
            # Clockwise
            current_pos[4] -= self.rotation_constant * math.pi / 180
        elif action == 5:
            # Counter-Clockwise
            current_pos[4] += self.rotation_constant * math.pi / 180
        else:
            logging.error("Unknown action: %i" % action)

        #print("Move to Pose: ", current_pos)

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

        state, terminal = self.observe()

        return state, reward, terminal

    def observe(self):
        return np.random.rand(50, 50), self.in_terminal_state()

    def in_terminal_state(self):
        if self.debug_nonterminal:
            self.debug_nonterminal = False
            return False
        else:
            return self.is_touching_wire() | self.is_in_dead_zone() | \
                   self.is_at_goal()

    def is_touching_wire(self):
        # todo: get wire/loop circuit signal
        wire_signal = np.random.randint(2)

        #if wire_signal:
        #    print("Loop is touching the wire!")

        return wire_signal

    def is_at_goal(self):
        if np.random.random() < .1:
            return True
        else:
            return False

    def is_in_dead_zone(self):
        if np.random.random() < .02:
            return True
        else:
            return False

    def invert_game(self):
        pass

    def advance_checkpoints(self):
        pass

    def regress_checkpoints(self):
        pass

    def is_at_old_checkpoint(self):
        pass

    def is_at_new_checkpoint(self):
        pass

    def reset(self):
        pass

    def engage(self):
        pass

    def disengage(self):
        pass
