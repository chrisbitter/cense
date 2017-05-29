# from Cense.World.world import World
from Cense.World.Camera.camera_videocapture import Camera
from Cense.World.Robot.rtde_controller import RTDE_Controller, IllegalPoseException
from Cense.World.Loop.loop import Loop
import math
import numpy as np
import logging


class TerminalStateError(Exception):
    def __init__(self, *args):
        super().__init__(self, *args)


class RealWorld(object):
    Y_DISENGAGED = -.215
    Y_ENGAGED = -.245

    # 0.34867198 -0.32503722  0.482323
    START_POSE = np.array([.345, Y_ENGAGED, .482, 0, 0, 0])
    # -0.215311   -0.32231305  0.50007683
    GOAL_POSE = np.array([-.215, Y_ENGAGED, .503, 0, 0, 0])

    CHECKPOINT_DISTANCE = .01
    DISTANCE_THRESHOLD = .003

    PUNISHMENT_ILLEGAL_POSE = -5
    PUNISHMENT_WIRE = -30
    PUNISHMENT_OLD_CHECKPOINT = -15
    REWARD_GOAL = 50
    REWARD_NEW_CHECKPOINT = 15
    REWARD_GENERIC = -1

    STATE_DIMENSIONS = (50, 50)
    ACTIONS = 6

    __checkpoints = []

    translation_constant = .005
    rotation_constant = 30
    camera = None

    loop = None

    last_action = None

    def __init__(self):
        self.controller = RTDE_Controller()
        self.camera = Camera()
        self.loop = Loop()

        # reset to START_POSE
        self.reset()

    def execute(self, action):

        # only move when state is not terminal
        if not self.in_terminal_state():

            next_pose = self.controller.current_pose()

            # all movements relative to TCP orientation
            if action == 0:
                # move in positive x
                next_pose[0] += self.translation_constant * math.cos(next_pose[4])
                next_pose[2] -= self.translation_constant * math.sin(next_pose[4])
            elif action == 1:
                # move in negative x
                next_pose[0] -= self.translation_constant * math.cos(next_pose[4])
                next_pose[2] += self.translation_constant * math.sin(next_pose[4])
            elif action == 2:
                # move in positive z
                next_pose[0] += self.translation_constant * math.sin(next_pose[4])
                next_pose[2] += self.translation_constant * math.cos(next_pose[4])
            elif action == 3:
                # move in negative z
                next_pose[0] -= self.translation_constant * math.sin(next_pose[4])
                next_pose[2] -= self.translation_constant * math.cos(next_pose[4])
            elif action == 4:
                # turn positively around y
                next_pose[4] += self.rotation_constant * math.pi / 180
            elif action == 5:
                # turn negatively around y
                next_pose[4] -= RealWorld.rotation_constant * math.pi / 180
            else:
                logging.error("Unknown action: %i" % action)

            terminal = False

            try:
                self.controller.move_to_pose(next_pose)
                self.last_action = action

                if self.loop.has_touched_wire():
                    reward = self.PUNISHMENT_WIRE
                    terminal = True

                elif self.is_at_old_checkpoint():
                    reward = self.PUNISHMENT_OLD_CHECKPOINT
                    self.regress_checkpoints()

                elif self.is_at_goal():
                    reward = self.REWARD_GOAL
                    self.reset()
                    terminal = True

                elif self.is_at_new_checkpoint():
                    reward = self.REWARD_NEW_CHECKPOINT
                    self.advance_checkpoints()
                else:
                    reward = self.REWARD_GENERIC

                state = self.observe_state()

            except IllegalPoseException:
                print("Execution of action", action, "would result in Illegal Pose!")
                reward = self.PUNISHMENT_ILLEGAL_POSE
                terminal = True
                state = self.observe_state()

        else:
            raise TerminalStateError("Cannot perform actions in terminal states!")

        return state, reward, terminal

    def observe_state(self):
        return self.camera.capture_image()

    def in_terminal_state(self):
        return self.loop.has_touched_wire() | \
               self.is_at_goal()

    def is_at_goal(self):
        current_pose = self.controller.current_pose()
        return np.linalg.norm(current_pose[:3] - self.GOAL_POSE[:3]) < self.DISTANCE_THRESHOLD

    def is_at_old_checkpoint(self):
        if len(self.__checkpoints) > 1:
            current_pose = self.controller.current_pose()

            for i in range(len(self.__checkpoints) - 2, -1, -1):
                if np.linalg.norm(current_pose[:3] - self.__checkpoints[i][:3]) < self.DISTANCE_THRESHOLD:
                    return True
        return False

    def is_at_new_checkpoint(self):
        current_pose = self.controller.current_pose()
        return np.linalg.norm(current_pose[:3] - self.__checkpoints[-1][:3]) < self.DISTANCE_THRESHOLD

    def regress_checkpoints(self):
        self.__checkpoints.pop()

    def advance_checkpoints(self):
        current_pose = self.controller.current_pose()
        self.__checkpoints.append(current_pose[:3])

    def invert_game(self):
        raise NotImplementedError

        logging.debug("invert_game")

        #self.disengage()

        pose = self.controller.current_pose()

        if pose[4] < math.pi:
            pose[4] += math.pi
        else:
            pose[4] -= math.pi

        self.controller.move_to_pose(pose)

        #self.engage()

        # switch start and goal
        temp = self.START_POSE
        self.START_POSE = self.GOAL_POSE
        self.GOAL_POSE = temp

        # reset checkpoints
        self.__checkpoints = [self.START_POSE]

    def init_nonterminal_state(self):
        print("init_nonterminal_state")
        self.reset()

    def reset(self):
        logging.debug("RealWorld reset")

        pose = self.controller.current_pose()
        pose[1] = self.Y_DISENGAGED
        self.controller.move_to_pose(pose)

        pose = self.START_POSE
        pose[1] = self.Y_DISENGAGED
        self.controller.move_to_pose(pose)

        # might have touched wire when disengaging
        self.loop.touched_wire = False

        pose[1] = self.Y_ENGAGED
        self.controller.move_to_pose(pose)

        self.__checkpoints = [self.START_POSE[:3]]

    # def engage(self):
    #     logging.debug("RealWorld engage")
    #     pose = self.controller.current_pose()
    #     pose[1] = self.Y_ENGAGED
    #     self.controller.move_to_pose(pose)
    #
    # def disengage(self):
    #     logging.debug("RealWorld disengage")
    #     pose = self.controller.current_pose()
    #     pose[1] = self.Y_DISENGAGED
    #     self.controller.move_to_pose(pose)

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
        print("Removed")
        # self.engage()

        input("Test disengage\nPRESS ENTER")
        print("Removed")
        # self.disengage()

    def test_observation(self):
        input("Test Observation")

        input("Test observe\nPRESS ENTER")
        state, terminal = self.observe_state()

        print("state\n", state, "\n")
        print("terminal: ", terminal)

        input("Test is_touching_wire\nPRESS ENTER")
        print(self.has_touched_wire())


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.ERROR)

    world = RealWorld()
