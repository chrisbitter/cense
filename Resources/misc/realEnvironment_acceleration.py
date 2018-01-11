import logging

import numpy as np
from copy import deepcopy

from Cense.Environment.Camera.camera import Camera as Camera
from Cense.Environment.Robot.rtdeController import RtdeController as Controller, IllegalPoseException, SpawnedInTerminalStateError, ExitedInTerminalStateError


class InsufficientProgressError(Exception):
    def __init__(self, *args):
        super(InsufficientProgressError, self).__init__(*args)

class UntreatableStateError(Exception):
    def __init__(self, *args):
        super(UntreatableStateError, self).__init__(*args)


class RealEnvironment(object):
    Y_DISENGAGED = -.3
    Y_ENGAGED = -.35

    START_POSE = np.array([.3, Y_ENGAGED, .458, 0, np.pi / 2, 0])
    #PREVIOUS_START_POSE = START_POSE

    CURRENT_START_POSE = START_POSE
    CURRENT_START_VELOCITY = np.zeros(3)

    GOAL_X = Controller.CONSTRAINT_MIN[0] + .03
    # GOAL_POSE = np.array([-.215, Y_ENGAGED, .503, 0, np.pi/2, 0])

    STATE_DIMENSIONS = (50, 50)
    VELOCITY_DIMENSIONS = (3,)
    ACTIONS = 27

    __checkpoints = []

    CURRENT_STEP_WATCHDOG = 0

    camera = None

    last_action = None

    velocity = np.zeros(3)

    def __init__(self, environment_config, set_status_func):

        self.CHECKPOINT_DISTANCE = environment_config["checkpoint_distance"]

        self.PUNISHMENT_WIRE = environment_config["punishment_wire"]
        self.PUNISHMENT_INSUFFICIENT_PROGRESS = environment_config["punishment_insufficient_progress"]
        self.PUNISHMENT_OLD_CHECKPOINT = environment_config["punishment_old_checkpoint"]
        self.REWARD_GOAL = environment_config["reward_goal"]
        self.REWARD_NEW_CHECKPOINT = environment_config["reward_new_checkpoint"]
        self.REWARD_GENERIC = environment_config["reward_generic"]

        self.STEP_WATCHDOG = environment_config["step_watchdog"]

        self.TRANSLATION_ACCELERATION_FORWARD = environment_config["translation_acceleration_forward"]
        self.TRANSLATION_ACCELERATION_SIDEWAYS = environment_config["translation_acceleration_sideways"]
        self.ROTATION_ACCELERATION = environment_config["rotation_acceleration"] * np.pi / 180

        self.MIN_TRANSLATION_VELOCITY_FORWARD = environment_config["min_translation_velocity_forward"]
        self.MAX_TRANSLATION_VELOCITY_FORWARD = environment_config["max_translation_velocity_forward"]

        self.MAX_ABS_TRANSLATION_VELOCITY_SIDEWAYS = environment_config["max_abs_translation_velocity_sideways"]
        self.MAX_ABS_ROTATION_VELOCITY = environment_config["max_abs_rotation_velocity"] * np.pi / 180

        self.set_status_func = set_status_func

        logging.debug("Real Environment - init")

        self.set_status_func("Setup Environment")

        self.controller = Controller(set_status_func)
        self.camera = Camera(self.STATE_DIMENSIONS, set_status_func)

        self.reset()

        self.reset_stepwatchdog()

    def reset_stepwatchdog(self):
        self.CURRENT_STEP_WATCHDOG = 0

    def execute(self, action):
        logging.debug("Real Environment - execute")

        self.CURRENT_STEP_WATCHDOG += 1

        # deep copy velocity
        old_velocity = np.empty_like(self.velocity)
        np.copyto(old_velocity, self.velocity)

        # calculate new velocity [mm/step]
        # 27 actions = 3x3x3
        # 0-8: accelerate forward
        # 9-17: brake forward
        # 18-26: do nothing forward

        # each block has 9 entries. For each:
        # 0-2: accelerate right
        # 3-5: accelerate left
        # 6-8: do nothing sideways

        # each subblock has 3 entries. For each:
        # 0: accelerate rotation right
        # 1: accelerate rotation left
        # 2: do nothing rotation

        # forward
        if action // 9 == 0:
            self.velocity[0] += self.TRANSLATION_ACCELERATION_FORWARD
        elif action // 9 == 1:
            self.velocity[0] -= self.TRANSLATION_ACCELERATION_FORWARD
        self.velocity[0] = max(min(self.velocity[0], self.MAX_TRANSLATION_VELOCITY_FORWARD),
                                 self.MIN_TRANSLATION_VELOCITY_FORWARD)

        # sideways
        if (action % 9) // 3 == 0:
            self.velocity[1] += self.TRANSLATION_ACCELERATION_SIDEWAYS
        elif (action % 9) // 3 == 1:
            self.velocity[1] -= self.TRANSLATION_ACCELERATION_SIDEWAYS
        self.velocity[1] = max(min(self.velocity[1], self.MAX_ABS_TRANSLATION_VELOCITY_SIDEWAYS),
                                 -self.MAX_ABS_TRANSLATION_VELOCITY_SIDEWAYS)

        if action % 3 == 0:
            self.velocity[2] += self.ROTATION_ACCELERATION
        elif action % 3 == 1:
            self.velocity[2] -= self.ROTATION_ACCELERATION
        self.velocity[2] = max(min(self.velocity[2], self.MAX_ABS_ROTATION_VELOCITY),
                                 -self.MAX_ABS_ROTATION_VELOCITY)

        # new pose:
        next_pose, _ = self.controller.current_pose()

        # forward
        next_pose[0] -= self.velocity[0] * np.sin(next_pose[4])
        next_pose[2] -= self.velocity[0] * np.cos(next_pose[4])
        # right (negative is left)
        next_pose[0] += self.velocity[1] * np.cos(next_pose[4])
        next_pose[2] -= self.velocity[1] * np.sin(next_pose[4])
        # rotation right (negative is left)
        next_pose[4] -= self.velocity[2]

        terminal = False

        try:
            touched_wire = self.controller.move_to_pose(next_pose)

            if touched_wire:
                reward = self.PUNISHMENT_WIRE
                terminal = True
                self.reset_stepwatchdog()

                if np.random.random() < .7:
                    # restore velocity to avoid that the robot accelerates up to maximum velocity
                    np.copyto(self.velocity, old_velocity)
                else:
                    self.velocity = np.zeros(3)

            elif self.is_at_goal():
                reward = self.REWARD_GOAL
                terminal = True
                self.reset_stepwatchdog()
                self.reset_current_start_pose()

            elif self.is_at_old_checkpoint():
                reward = self.PUNISHMENT_OLD_CHECKPOINT
                self.regress_checkpoints()
                self.reset_stepwatchdog()
            elif self.is_at_new_checkpoint():
                reward = self.REWARD_NEW_CHECKPOINT
                self.advance_checkpoints()
                self.reset_stepwatchdog()
            else:
                reward = -self.CURRENT_STEP_WATCHDOG/self.STEP_WATCHDOG
                #reward = self.REWARD_GENERIC

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

    def get_normalized_velocity(self):
        normalized_velocity = [self.velocity[0] / self.MAX_TRANSLATION_VELOCITY_FORWARD,
                                 self.velocity[1] / self.MAX_ABS_TRANSLATION_VELOCITY_SIDEWAYS,
                                 self.velocity[2] / self.MAX_ABS_ROTATION_VELOCITY]

        return normalized_velocity

    def observe_state(self):
        logging.debug("Real Environment - observe_state")

        return [self.camera.capture_image(), self.get_normalized_velocity()]

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
                   2*np.linalg.norm(current_pose[:3] - self.__checkpoints[-2][:3]) and not touching_wire

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

        self.velocity = deepcopy(self.CURRENT_START_VELOCITY)

        pose, _ = self.controller.current_pose()
        pose[1] = self.Y_DISENGAGED
        try:
            self.controller.move_to_pose(pose, force=True)
        except IllegalPoseException:
            raise

        pose = deepcopy(self.CURRENT_START_POSE)

        pose[1] = self.Y_DISENGAGED
        try:
            self.controller.move_to_pose(pose, force=True)
        except IllegalPoseException:
            raise

        pose[1] = self.Y_ENGAGED

        reset_failed = False

        try:
            touched_wire = self.controller.move_to_pose(pose)

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
            #self.PREVIOUS_START_POSE = self.CURRENT_START_POSE
            self.CURRENT_START_POSE = current_pose
            self.CURRENT_START_VELOCITY = deepcopy(self.velocity)
        else:
            print("Not updating start pose because loop is touching the wire")

    def reset_current_start_pose(self):
        logging.debug("Real Environment - reset_current_start_pose")
        #self.PREVIOUS_START_POSE = self.CURRENT_START_POSE
        self.CURRENT_START_POSE = deepcopy(self.START_POSE)
        self.CURRENT_START_VELOCITY = np.zeros(3)


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

    world = RealEnvironment(config, print)

    #world.execute(3)
