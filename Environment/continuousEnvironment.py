import logging

import numpy as np

from Environment.Camera.camera import Camera as Camera
from Environment.Robot.rtdeController import RtdeController as Controller, IllegalPoseException, \
    SpawnedInTerminalStateError, ExitedInTerminalStateError



class InsufficientProgressError(Exception):
    def __init__(self, *args):
        super(InsufficientProgressError, self).__init__(*args)


class UntreatableStateError(Exception):
    def __init__(self, *args):
        super(UntreatableStateError, self).__init__(*args)


class ContinuousEnvironment(object):

    #####################################################################
    ##### Parameters

    ###Physical
    Y_DISENGAGED = -.32
    Y_ENGAGED = -.38

    START_POSE = np.array([.27, Y_ENGAGED, .38, 0, -np.pi/2, 0])

    GOAL_X = -.23
    #GOAL_X = Controller.CONSTRAINT_MIN[0] + .03

    ###Reinforcement Learning
    STATE_DIMENSIONS = (40, 40, 3)
    ACTIONS = 3

    #####################################################################

    CURRENT_START_POSE = START_POSE

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

        self.TRANSLATION_FORWARD_MAX_DISTANCE = environment_config["translation_forward_max_distance"]
        self.TRANSLATION_SIDEWAYS_MAX_DISTANCE = environment_config["translation_sideways_max_distance"]
        self.ROTATION_MAX_ANGLE = environment_config["rotation_max_angle"] * np.pi / 180
        if "start_pose" in environment_config:
            assert len(environment_config["start_pose"]) == 6
            global CURRENT_START_POSE, Y_ENGAGED
            CURRENT_START_POSE = np.array(environment_config["start_pose"])
            CURRENT_START_POSE[4] *= np.pi / 180
            Y_ENGAGED = CURRENT_START_POSE[1]

        logging.debug("Real Environment - init")

        print("Setup Environment")

        self.controller = Controller()
        self.camera = Camera(self.STATE_DIMENSIONS)

        self.reset()

        self.reset_stepwatchdog()

    def reset_stepwatchdog(self):
        self.CURRENT_STEP_WATCHDOG = 0

    def execute(self, action):
        logging.debug("Real Environment - execute")

        # temp = action[0]
        # action[0] = action[1]
        # action[1] = temp

        self.CURRENT_STEP_WATCHDOG += 1

        next_pose, _ = self.controller.current_pose()

        # all movements relative to TCP orientation

        next_pose[0] += action[0] * self.TRANSLATION_SIDEWAYS_MAX_DISTANCE * np.sin(next_pose[4]) \
                        + action[1] * self.TRANSLATION_FORWARD_MAX_DISTANCE * np.cos(next_pose[4])
        next_pose[2] += action[0] * self.TRANSLATION_SIDEWAYS_MAX_DISTANCE * np.cos(next_pose[4]) \
                        - action[1] * self.TRANSLATION_FORWARD_MAX_DISTANCE * np.sin(next_pose[4])
        next_pose[4] += action[2] * self.ROTATION_MAX_ANGLE

        terminal = False

        try:
            touched_wire, mean_percentage_traveled = self.controller.move_to_pose(next_pose)

            if touched_wire:
                reward = self.PUNISHMENT_WIRE * (1 - .4 * mean_percentage_traveled)
                terminal = True
                self.CURRENT_STEP_WATCHDOG -= 1
                # self.reset_stepwatchdog()
            elif self.is_at_goal():
                reward = self.REWARD_GOAL
                terminal = True
                self.reset_stepwatchdog()
            elif self.is_at_old_checkpoint():
                reward = self.PUNISHMENT_OLD_CHECKPOINT
                self.regress_checkpoints()
                self.reset_stepwatchdog()
            elif self.is_at_new_checkpoint():
                reward = self.REWARD_NEW_CHECKPOINT
                self.advance_checkpoints()
                self.reset_stepwatchdog()
            else:
                reward = .8 * (self.CURRENT_STEP_WATCHDOG / self.STEP_WATCHDOG) * self.PUNISHMENT_INSUFFICIENT_PROGRESS \
                         + .2 * action[0]

            # print(action, reward, mean_percentage_traveled)

            if self.CURRENT_STEP_WATCHDOG >= self.STEP_WATCHDOG:
                self.reset_stepwatchdog()
                raise InsufficientProgressError

            state = self.observe_state()

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

        return self.camera.get_frame()

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

        # print(self.DIFF_BETA)
        #
        # while self.DIFF_BETA > .6*np.pi:
        #     print("sub")
        #     pose[4] -= np.pi / 2
        #     self.DIFF_BETA -= np.pi / 2
        #     self.controller.move_to_pose(pose, force=True)
        #
        # while self.DIFF_BETA < -.6*np.pi:
        #     print("add")
        #     pose[4] += np.pi / 2
        #     self.DIFF_BETA += np.pi / 2
        #     self.controller.move_to_pose(pose, force=True)
        #
        # self.DIFF_BETA = self.CURRENT_START_DIFF_BETA

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
            # self.CURRENT_START_DIFF_BETA = self.DIFF_BETA
        else:
            print("Not updating start pose because loop is touching the wire")

    def reset_current_start_pose(self):
        logging.debug("Real Environment - reset_current_start_pose")
        # self.PREVIOUS_START_POSE = self.CURRENT_START_POSE
        self.CURRENT_START_POSE = self.START_POSE
        # self.CURRENT_START_DIFF_BETA = self.START_DIFF_BETA


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

        "translation_forward_max_distance": 0.03,
        "translation_sideways_max_distance": 0.03,
        "rotation_max_angle": 90
    }

    world = ContinuousEnvironment(config)

    world.execute([0.2,0,0])
    world.execute([0,.2,0])
    world.execute([0,-.2,0])
    world.execute([0,0,.2])
    world.execute([0,0,-.2])

    # for i in range(3):
    #     world.execute([0, 0, -1])
    #
    # world.reset()
