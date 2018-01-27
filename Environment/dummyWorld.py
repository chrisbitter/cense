# from Cense.World.world import World
# import RTDE_Controller_CENSE as rtde

import numpy as np

from Environment.Camera.camera_pygame import Camera


class TerminalStateError(Exception):
    def __init__(self, *args):
        super().__init__(self, *args)


class InsufficientProgressError(Exception):
    def __init__(self, *args):
        super(InsufficientProgressError, self).__init__(*args)


class UntreatableStateError(Exception):
    def __init__(self, *args):
        super(UntreatableStateError, self).__init__(*args)


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

    STATE_DIMENSIONS = (50, 50,)
    ACTIONS = 3

    __checkpoints = []
    camera = None

    debug_enter_nonterminal = False
    terminal_probability = .6

    translation_constant = .01
    rotation_constant = 45

    def __init__(self, config, status):
        self.camera = Camera()

    def init_nonterminal_state(self):
        self.debug_nonterminal = True

    def execute(self, action):

        current_pos = np.zeros(6)

        print(action)

        # print("Move to Pose: ", current_pos)

        terminal = False

        if self.is_touching_wire():
            reward = self.PUNISHMENT_WIRE
            terminal = True

        elif self.is_in_dead_zone():
            reward = self.PUNISHMENT_DEAD_ZONE
            terminal = True

        elif self.is_at_old_checkpoint():
            reward = self.PUNISHMENT_OLD_CHECKPOINT
            self.regress_checkpoints()

        elif self.is_at_goal():
            reward = self.REWARD_GOAL
            terminal = True

        elif self.is_at_new_checkpoint():
            reward = self.REWARD_NEW_CHECKPOINT
            self.advance_checkpoints()
        else:
            reward = self.REWARD_GENERIC

        state = self.observe_state()

        return state, reward, terminal

    def observe_state(self):
        return np.random.rand(50, 50)

    def update_current_start_pose(self):
        pass

    def in_terminal_state(self):
        if np.random.random() < self.terminal_probability:
            self.terminal_probability = max(.05, self.terminal_probability * .9)
            return True
        else:
            return False

    def is_touching_wire(self):
        # todo: get wire/loop circuit signal
        wire_signal = np.random.randint(2)

        # if wire_signal:
        #    print("Loop is touching the wire!")

        return wire_signal

    def is_at_goal(self):
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

    def reset(self, hard_reset=False):
        pass

    def engage(self):
        pass

    def disengage(self):
        pass
