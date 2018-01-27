#Todo: remove discreteEnvironment by moving to continuousEnvironment

import logging

import numpy as np

from Environment import Camera as Camera
from Environment import RtdeController as Controller, IllegalPoseException, \
    SpawnedInTerminalStateError, ExitedInTerminalStateError


class InsufficientProgressError(Exception):
    def __init__(self, *args):
        super(InsufficientProgressError, self).__init__(*args)


class UntreatableStateError(Exception):
    def __init__(self, *args):
        super(UntreatableStateError, self).__init__(*args)


class DiscreteEnvironment(object):
    Y_DISENGAGED = -.3
    Y_ENGAGED = -.35

    START_POSE = np.array([.3, Y_ENGAGED, .448, 0, np.pi / 2, 0])
    # PREVIOUS_START_POSE = START_POSE

    CURRENT_START_POSE = START_POSE
    GOAL_X = Controller.CONSTRAINT_MIN[0] + .03
    # GOAL_POSE = np.array([-.215, Y_ENGAGED, .503, 0, np.pi/2, 0])

    STATE_DIMENSIONS = (40, 40)
    ACTIONS = 5

    __checkpoints = []

    CURRENT_STEP_WATCHDOG = 0

    camera = None

    last_action = None

    def __init__(self, environment_config):

        self.CHECKPOINT_DISTANCE = environment_config["checkpoint_distance"]

        self.PUNISHMENT_WIRE = environment_config["punishment_wire"]
        self.PUNISHMENT_INSUFFICIENT_PROGRESS = environment_config["punishment_insufficient_progress"]
        self.PUNISHMENT_OLD_CHECKPOINT = environment_config["punishment_old_checkpoint"]
        self.REWARD_GOAL = environment_config["reward_goal"]
        self.REWARD_NEW_CHECKPOINT = environment_config["reward_new_checkpoint"]
        self.REWARD_GENERIC = environment_config["reward_generic"]

        self.STEP_WATCHDOG = environment_config["step_watchdog"]

        self.TRANSLATION_DISTANCE = environment_config["translation_distance"]
        self.ROTATION_ANGLE = environment_config["rotation_angle"] * np.pi / 180

        logging.debug("Real Environment - init")

        print("Setup Environment")

        self.controller = Controller()
        self.camera = Camera(self.STATE_DIMENSIONS)

        self.reset()

        self.reset_stepwatchdog()

    def reset_stepwatchdog(self):
        self.CURRENT_STEP_WATCHDOG = 0

    def execute(self, action, scale=1):
        logging.debug("Real Environment - execute")

        self.CURRENT_STEP_WATCHDOG += 1

        next_pose, _ = self.controller.current_pose()

        # all movements relative to TCP orientation
        # backwards movement disabled
        if action == 0:
            # move left
            next_pose[0] -= scale * self.TRANSLATION_DISTANCE * np.cos(next_pose[4])
            next_pose[2] += scale * self.TRANSLATION_DISTANCE * np.sin(next_pose[4])
        elif action == 1:
            # rotate left
            next_pose[4] += scale * self.ROTATION_ANGLE
        elif action == 2:
            # move forward
            next_pose[0] -= scale * self.TRANSLATION_DISTANCE * np.sin(next_pose[4])
            next_pose[2] -= scale * self.TRANSLATION_DISTANCE * np.cos(next_pose[4])
        elif action == 3:
            # rotate right
            next_pose[4] -= scale * self.ROTATION_ANGLE
        elif action == 4:
            # move right
            next_pose[0] += scale * self.TRANSLATION_DISTANCE * np.cos(next_pose[4])
            next_pose[2] -= scale * self.TRANSLATION_DISTANCE * np.sin(next_pose[4])
        else:
            logging.error("Unknown action: %i" % action)

        terminal = False

        try:
            touched_wire, _ = self.controller.move_to_pose(next_pose)

            if touched_wire:
                # print("touched")
                reward = self.PUNISHMENT_WIRE
                terminal = True
                self.CURRENT_STEP_WATCHDOG -= 1
            elif self.is_at_goal():
                # print("goal")
                reward = self.REWARD_GOAL
                terminal = True
                self.reset_stepwatchdog()
            elif self.is_at_old_checkpoint():
                # print("old")
                reward = self.PUNISHMENT_OLD_CHECKPOINT
                self.regress_checkpoints()
                self.reset_stepwatchdog()
            elif self.is_at_new_checkpoint():
                # print("new")
                reward = self.REWARD_NEW_CHECKPOINT
                self.advance_checkpoints()
                self.reset_stepwatchdog()
            else:
                # print("generic")
                reward = -self.CURRENT_STEP_WATCHDOG / self.STEP_WATCHDOG

            if self.CURRENT_STEP_WATCHDOG >= self.STEP_WATCHDOG:
                self.reset_stepwatchdog()
                raise InsufficientProgressError

            state = self.observe_state()

            # print(reward)

            return state, reward, terminal

        except IllegalPoseException:
            raise
        except (SpawnedInTerminalStateError, ExitedInTerminalStateError):
            try:
                self.reset()
            except UntreatableStateError:
                raise
            raise

    def observe_state(self):
        logging.debug("Real Environment - observe_state")

        return self.camera.capture_image()

    def is_at_goal(self):
        logging.debug("Real Environment - is_at_goal")

        current_pose, _ = self.controller.current_pose()
        return current_pose[0] < self.GOAL_X

    def is_at_new_checkpoint(self):
        logging.debug("Real Environment - is_at_new_checkpoint")
        current_pose, touching_wire = self.controller.current_pose()
        return np.linalg.norm(
            current_pose[:3] - self.__checkpoints[-1][:3]) > self.CHECKPOINT_DISTANCE and not touching_wire

    def is_at_old_checkpoint(self):
        # considered at old checkpoint, if distance to current checkpoint is bigger than
        #  2x the distance to the old checkpoint
        logging.debug("Real Environment - is_at_old_checkpoint")
        current_pose, touching_wire = self.controller.current_pose()
        if len(self.__checkpoints) > 1:
            return np.linalg.norm(current_pose[:3] - self.__checkpoints[-1][:3]) > \
                   2 * np.linalg.norm(current_pose[:3] - self.__checkpoints[-2][:3]) and not touching_wire

    def advance_checkpoints(self):
        logging.debug("Real Environment - advance_checkpoints")

        current_pose, touching_wire = self.controller.current_pose()
        if not touching_wire:
            self.__checkpoints.append(current_pose[:3])
        else:
            logging.info("Not advancing checkpoints because loop is touching the wire")

    def regress_checkpoints(self):
        if len(self.__checkpoints) > 1:
            self.__checkpoints.pop()

    def reset(self, hard_reset=False):
        logging.debug("Real Environment - reset")

        if hard_reset:
            self.reset_current_start_pose()

        pose, _ = self.controller.current_pose()
        pose[1] = self.Y_DISENGAGED
        try:
            self.controller.move_to_pose(pose, force=True)
        except IllegalPoseException:
            raise

        pose = self.CURRENT_START_POSE
        pose[1] = self.Y_DISENGAGED
        try:
            self.controller.move_to_pose(pose, force=True)
        except IllegalPoseException:
            raise

        pose[1] = self.Y_ENGAGED

        reset_failed = False

        try:
            touched_wire, _ = self.controller.move_to_pose(pose)

            if touched_wire:
                reset_failed = True

        except (SpawnedInTerminalStateError, ExitedInTerminalStateError):
            reset_failed = True
        except IllegalPoseException:
            raise

        if reset_failed:
            if (self.CURRENT_START_POSE != self.START_POSE).any():
                # try hard reset
                self.reset(hard_reset=True)
            else:
                # reset didn't manage to get to a nonterminal state, something went seriously wrong!
                raise UntreatableStateError

        # reset checkpoints
        self.__checkpoints = [self.CURRENT_START_POSE[:3]]

        self.reset_stepwatchdog()

    def update_current_start_pose(self):
        logging.debug("Real Environment - update_current_start_pose")

        current_pose, touching_wire = self.controller.current_pose()

        if not touching_wire:
            # self.PREVIOUS_START_POSE = self.CURRENT_START_POSE
            self.CURRENT_START_POSE = current_pose
        else:
            print("Not updating start pose because loop is touching the wire")

    def reset_current_start_pose(self):
        logging.debug("Real Environment - reset_current_start_pose")
        # self.PREVIOUS_START_POSE = self.CURRENT_START_POSE
        self.CURRENT_START_POSE = self.START_POSE


if __name__ == "__main__":
    # logging.getLogger().setLevel(logging.DEBUG)

    config = {

        "checkpoint_distance": 0.03,

        "punishment_wire": -1,
        "punishment_insufficient_progress": -0.5,
        "punishment_old_checkpoint": -0.5,
        "reward_goal": 1,
        "reward_new_checkpoint": 1,
        "reward_generic": -0.1,

        "step_watchdog": 10,

        "translation_distance": 0.01,
        "rotation_angle": 30
    }

    world = DiscreteEnvironment(config)

    # world.execute(3)
