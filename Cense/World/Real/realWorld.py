# from Cense.World.world import World
from Cense.World.Camera.camera import Camera
import RTDE_Controller_CENSE as rtde
import math
import numpy as np
import logging


class TerminalStateError(Exception):
    def __init__(self, *args):
        super().__init__(self, *args)


class RealWorld(object):
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

    STATE_DIMENSIONS = (50, 50)
    ACTIONS = 6

    __checkpoints = []
    camera = None

    translation_constant = .01
    rotation_constant = 45
    camera = None

    def __init__(self):
        self.camera = Camera()

    def execute(self, action):
        # only move when state is not terminal
        if not self.in_terminal_state():

            current_pos = rtde.current_position()

            if len(action) > 1:
                action = np.argmax(action)

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
                current_pos[4] += RealWorld.rotation_constant * math.pi / 180
            else:
                logging.error("Unknown action: %i" % action)

            rtde.move_to_position(current_pos)

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
            raise TerminalStateError("Cannot perform actions in terminal states!")

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

        self.disengage()
        rtde.rotate_180()
        self.engage()

        # switch start and goal
        temp = self.START_POSE
        self.START_POSE = self.GOAL_POSE
        self.GOAL_POSE = temp

        # reset checkpoints
        self.__checkpoints = [self.START_POSE]

    def reset(self):
        logging.debug("RealWorld reset")
        pose = self.START_POSE
        pose[2] = self.Z_DISENGAGED
        rtde.move_to_position(pose)

    def engage(self):
        logging.debug("RealWorld engage")
        pose = rtde.current_position()
        pose[2] = self.Z_ENGAGED
        rtde.move_to_position(pose)

    def disengage(self):
        logging.debug("RealWorld disengage")
        pose = rtde.current_position()
        pose[2] = self.Z_DISENGAGED
        rtde.move_to_position(pose)

    def test_movement(self):
        input("Test Movement")

        input("Test reset\nPRESS ENTER")

        self.reset()

        for i in range(6):
            action = np.zeros(1, 6)
            input("Test action %s\nPRESS ENTER" % str(action))
            action[i] = 1
            self.execute(action)

        input("Test engage\nPRESS ENTER")
        self.engage()

        input("Test disengage\nPRESS ENTER")
        self.disengage()

    def test_observation(self):
        input("Test Observation")

        input("Test observe\nPRESS ENTER")
        state, terminal = self.observe()

        print("state\n", state, "\n")
        print("terminal: ", terminal)

        input("Test is_touching_wire\nPRESS ENTER")
        print(self.is_touching_wire())


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)

    logging.info("Started RealWorld as main")
    world = RealWorld()

    world.test_movement()
    world.test_observation()

    print("Done")
