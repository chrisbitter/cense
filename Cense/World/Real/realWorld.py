import logging

import numpy as np

from Cense.World.Camera.camera_videocapture import Camera as Camera
from Cense.World.Robot.rtdeController import RtdeController as Controller, IllegalPoseException, TerminalStateError


class InsufficientProgressError(Exception):
    def __init__(self, *args):
        super(self, *args)


class RealWorld(object):
    Y_DISENGAGED = -.3
    Y_ENGAGED = -.35

    START_POSE = np.array([.3, Y_ENGAGED, .448, 0, np.pi / 2, 0])

    CURRENT_START_POSE = START_POSE
    GOAL_X = -.215
    # GOAL_POSE = np.array([-.215, Y_ENGAGED, .503, 0, np.pi/2, 0])

    CHECKPOINT_DISTANCE = .03
    DISTANCE_THRESHOLD = .015

    PUNISHMENT_ILLEGAL_POSE = -1
    PUNISHMENT_WIRE = -1
    REWARD_GOAL = 1
    REWARD_NEW_CHECKPOINT = 1
    REWARD_GENERIC = -.01

    STATE_DIMENSIONS = (40, 40)
    ACTIONS = 5

    __checkpoints = []

    STEP_WATCHDOG = 10
    CURRENT_STEP_WATCHDOG = 0

    translation_constant = .01
    translation_step_size = .002
    rotation_constant = 30 * np.pi / 180
    rotation_step_size = 10 * np.pi / 180

    camera = None

    last_action = None

    def __init__(self, set_status_func):

        self.set_status_func = set_status_func

        logging.debug("Real World - init")

        self.set_status_func("Setup World")

        self.controller = Controller(set_status_func)
        self.camera = Camera(self.STATE_DIMENSIONS, set_status_func)

        self.reset()

        self.reset_stepwatchdog()

    def reset_stepwatchdog(self):
        self.CURRENT_STEP_WATCHDOG = self.STEP_WATCHDOG

    def execute(self, action, force=False):
        logging.debug("Real World - execute")

        if not force:
            if self.CURRENT_STEP_WATCHDOG == 0:
                self.reset()
                raise InsufficientProgressError
            else:
                self.CURRENT_STEP_WATCHDOG -= 1

        next_pose, touching_wire = self.controller.current_pose()

        # all movements relative to TCP orientation
        # backwards movement disabled
        if action == 0:
            # move in positive x
            next_pose[0] += self.translation_constant * np.cos(next_pose[4])
            next_pose[2] -= self.translation_constant * np.sin(next_pose[4])
        elif action == 1:
            # move in negative x
            next_pose[0] -= self.translation_constant * np.cos(next_pose[4])
            next_pose[2] += self.translation_constant * np.sin(next_pose[4])
        elif action == -1:
            # move in positive z
            next_pose[0] += self.translation_constant * np.sin(next_pose[4])
            next_pose[2] += self.translation_constant * np.cos(next_pose[4])
        elif action == 2:
            # move in negative z
            next_pose[0] -= self.translation_constant * np.sin(next_pose[4])
            next_pose[2] -= self.translation_constant * np.cos(next_pose[4])
        elif action == 3:
            # turn positively around y
            next_pose[4] += self.rotation_constant
        elif action == 4:
            # turn negatively around y
            next_pose[4] -= RealWorld.rotation_constant
        else:
            logging.error("Unknown action: %i" % action)

        terminal = False

        try:
            touched_wire = self.controller.move_to_pose(next_pose, force)

            if touched_wire:
                reward = self.PUNISHMENT_WIRE
                terminal = True
            elif self.is_at_goal():
                reward = self.REWARD_GOAL
                terminal = True
            elif self.is_at_new_checkpoint():
                reward = self.REWARD_NEW_CHECKPOINT
                self.advance_checkpoints()
                self.reset_stepwatchdog()
            else:
                reward = self.REWARD_GENERIC

            # if state is terminal, it won't be used in DQN. So save memory by returning empty array
            state = []
            if not terminal:
                state = self.observe_state()

            return state, reward, terminal

        except IllegalPoseException:
            # should never occur with current setup!
            raise
        except TerminalStateError:
            try:
                self.reset()
                return None, None, None
            except:
                raise

    def observe_state(self):
        logging.debug("Real World - observe_state")

        return self.camera.capture_image()

    def is_at_goal(self):
        logging.debug("Real World - is_at_goal")

        current_pose, _ = self.controller.current_pose()
        return current_pose[0] < self.GOAL_X

    def is_at_new_checkpoint(self):
        logging.debug("Real World - is_at_new_checkpoint")
        current_pose, touching_wire = self.controller.current_pose()
        return np.linalg.norm(
            current_pose[:3] - self.__checkpoints[-1][:3]) > self.CHECKPOINT_DISTANCE and not touching_wire

    def advance_checkpoints(self):
        logging.debug("Real World - advance_checkpoints")

        current_pose, touching_wire = self.controller.current_pose()
        if not touching_wire:
            self.__checkpoints.append(current_pose[:3])
        else:
            logging.info("Not advancing checkpoints because loop is touching the wire")

    def reset(self, hard_reset=True):
        logging.debug("Real World - reset")

        if hard_reset:
            self.reset_current_start_pose()

        pose, _ = self.controller.current_pose()
        pose[1] = self.Y_DISENGAGED
        self.controller.move_to_pose(pose, force=True)

        pose = self.CURRENT_START_POSE
        pose[1] = self.Y_DISENGAGED
        self.controller.move_to_pose(pose, force=True)

        pose[1] = self.Y_ENGAGED
        try:
            touched_wire = self.controller.move_to_pose(pose)
        except TerminalStateError:
            touched_wire = True

        if touched_wire:
            # if current_start is not global start, try resetting everything
            if (self.CURRENT_START_POSE != self.START_POSE).all():

                # try resetting again with global start
                try:
                    self.reset(True)
                except:
                    # global pose couldn't be reached without touching the wire
                    raise
            else:
                raise TerminalStateError

        # reset checkpoints
        self.__checkpoints = [self.CURRENT_START_POSE[:3]]

        self.reset_stepwatchdog()

    def update_current_start_pose(self):
        logging.debug("Real World - update_current_start_pose")

        current_pose, touching_wire = self.controller.current_pose()

        if not touching_wire:
            self.CURRENT_START_POSE = current_pose
        else:
            logging.info("Not updating start pose because loop is touching the wire")

    def reset_current_start_pose(self):
        logging.debug("Real World - reset_current_start_pose")
        self.CURRENT_START_POSE = self.START_POSE


if __name__ == "__main__":
    # logging.getLogger().setLevel(logging.DEBUG)

    world = RealWorld(print)

    for _ in range(20):
        world.execute(4)
