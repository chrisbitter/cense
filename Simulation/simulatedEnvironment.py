import array
import Simulation.vrep as vrep
import time
import numpy as np


class IllegalPoseException(Exception):
    def __init__(self, *args):
        super(IllegalPoseException, self).__init__(*args)


class simulationEnvironment(object):
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

        self.START_POSE = np.array([.31, -.2, 1.12, -np.pi, -np.pi / 4, np.pi])

        self.CURRENT_START_POSE = self.START_POSE

        self.CONSTRAINT_MIN = np.array([-.35, -.25, .7])
        self.CONSTRAINT_MAX = np.array([.35, -.15, 1.3])

        self.ERROR_TRANSLATION = .001
        self.ERROR_ROTATION = 1 * np.pi / 180

        self.GOAL_X = -.25

        self.STATE_DIMENSIONS = (40, 40, 3)
        self.ACTIONS = 3
        self.PUNISHMENT_WIRE = -1
        self.PUNISHMENT_INSUFFICIENT_PROGRESS = -1

        self.CURRENT_STEP_WATCHDOG = 0

        vrep.simxFinish(-1)  # just in case, close all opened connections

        self.clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)

        if self.clientID != -1:
            _, self.targetHandle = vrep.simxGetObjectHandle(self.clientID, 'UR5_target#', vrep.simx_opmode_blocking)
            _, self.tipHandle = vrep.simxGetObjectHandle(self.clientID, 'UR5_tip', vrep.simx_opmode_blocking)
            _, self.collisionHandle = vrep.simxGetCollisionHandle(self.clientID, 'WireLoopCollision',
                                                                  vrep.simx_opmode_blocking)
            _, self.cameraHandle = vrep.simxGetObjectHandle(self.clientID, 'Camera', vrep.simx_opmode_blocking)

            vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_blocking)

            self.current_pose = self.CURRENT_START_POSE.copy()

            self.reset()

    def __del__(self):
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_blocking)

    def reset_stepwatchdog(self):
        self.CURRENT_STEP_WATCHDOG = 0

    def reset(self, hard_reset=False):

        if hard_reset:
            self.reset_current_start_pose()

        self.move_to_pose(self.CURRENT_START_POSE)

        vrep.simxSetIntegerSignal(self.clientID, 'Collision', 0, vrep.simx_opmode_blocking)

    def is_at_goal(self):

        position, _ = self.get_pose()

        return self.GOAL_X > position[0]

    def update_current_start_pose(self):
        pass

    def move_to_pose(self, next_pose):

        if next_pose[0] < self.CONSTRAINT_MIN[0] or next_pose[0] > self.CONSTRAINT_MAX[0] \
                or next_pose[1] < self.CONSTRAINT_MIN[1] or next_pose[1] > self.CONSTRAINT_MAX[1] \
                or next_pose[2] < self.CONSTRAINT_MIN[2] or next_pose[2] > self.CONSTRAINT_MAX[2]:
            print(next_pose)
            raise IllegalPoseException

        touched_wire = False

        while True:
            vrep.simxSetObjectPosition(self.clientID, self.targetHandle, -1, next_pose[:3], vrep.simx_opmode_oneshot)
            vrep.simxSetObjectOrientation(self.clientID, self.targetHandle, -1, next_pose[3:], vrep.simx_opmode_oneshot)

            current_tip_position, current_tip_orientation = self.get_pose()
            current_target_position, current_target_orientation = self.get_target_pose()

            translation_deviation = np.sum(np.absolute(current_tip_position - current_target_position))
            rotation_deviation = np.sum(np.absolute(
                ((np.array(current_tip_orientation - current_target_orientation) + np.pi) % (2 * np.pi)) - np.pi))

            if translation_deviation < self.ERROR_TRANSLATION and rotation_deviation < self.ERROR_ROTATION:
                break

            _, collision = vrep.simxGetIntegerSignal(self.clientID, "Collision", vrep.simx_opmode_blocking)

            touched_wire |= collision

        if touched_wire:
            self.reset()

        return touched_wire

    def get_pose(self):
        while True:
            returnCode, position = vrep.simxGetObjectPosition(self.clientID, self.tipHandle, -1,
                                                              vrep.simx_opmode_blocking)
            if returnCode == vrep.simx_return_ok:
                break

        while True:
            returnCode, orientation = vrep.simxGetObjectOrientation(self.clientID, self.tipHandle, -1,
                                                                    vrep.simx_opmode_blocking)
            if returnCode == vrep.simx_return_ok:
                break

        return np.array(position), np.array(orientation)

    def get_target_pose(self):
        while True:
            returnCode, position = vrep.simxGetObjectPosition(self.clientID, self.targetHandle, -1,
                                                              vrep.simx_opmode_blocking)
            if returnCode == vrep.simx_return_ok:
                break

        while True:
            returnCode, orientation = vrep.simxGetObjectOrientation(self.clientID, self.targetHandle, -1,
                                                                    vrep.simx_opmode_blocking)
            if returnCode == vrep.simx_return_ok:
                break

        return np.array(position), np.array(orientation)

    def observe_state(self):

        _, resolution, image = vrep.simxGetVisionSensorImage(self.clientID, self.cameraHandle, 0,
                                                             vrep.simx_opmode_blocking)

        state = np.array(image, dtype=bytes).reshape(tuple(resolution) + (3,)).astype(np.ubyte)

        state = (state / 127.5) - 1

        return state

    def execute(self, action):

        self.CURRENT_STEP_WATCHDOG += 1

        self.current_pose[0] -= action[0] * self.TRANSLATION_SIDEWAYS_MAX_DISTANCE * np.cos(
            self.current_pose[4]) - action[1] * self.TRANSLATION_FORWARD_MAX_DISTANCE * np.sin(self.current_pose[4])
        self.current_pose[2] += action[0] * self.TRANSLATION_SIDEWAYS_MAX_DISTANCE * np.sin(
            self.current_pose[4]) + action[1] * self.TRANSLATION_FORWARD_MAX_DISTANCE * np.cos(self.current_pose[4])

        self.current_pose[4] += action[2] * self.ROTATION_MAX_ANGLE
        self.current_pose[4] %= (2 * np.pi)

        # print(new_orientation[1])
        touched_wire = self.move_to_pose(self.current_pose)

        terminal = False

        if touched_wire:
            reward = -1
            terminal = True
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

        return self.observe_state(), reward, terminal

    def update_current_start_pose(self):
        current_pose, touching_wire = self.current_pose()

        if not touching_wire:
            # self.PREVIOUS_START_POSE = self.CURRENT_START_POSE
            self.CURRENT_START_POSE = current_pose
            # self.CURRENT_START_DIFF_BETA = self.DIFF_BETA
        else:
            print("Not updating start pose because loop is touching the wire")

    def reset_current_start_pose(self):
        self.CURRENT_START_POSE = self.START_POSE

    def is_at_new_checkpoint(self):
        return np.linalg.norm(
            self.current_position - self.__checkpoints[-1][:3]) > self.CHECKPOINT_DISTANCE

    def is_at_old_checkpoint(self):
        # considered at old checkpoint, if distance to current checkpoint is bigger than
        #  2x the distance to the old checkpoint
        current_pose, touching_wire = self.current_pose()
        if len(self.__checkpoints) > 1:
            return np.linalg.norm(current_pose[:3] - self.__checkpoints[-1][:3]) > \
                   2 * np.linalg.norm(current_pose[:3] - self.__checkpoints[-2][:3]) and not touching_wire

    def advance_checkpoints(self):

        current_pose, touching_wire = self.controller.current_pose()
        if not touching_wire:
            self.__checkpoints.append(current_pose[:3])

    def regress_checkpoints(self):
        if len(self.__checkpoints) > 1:
            self.__checkpoints.pop()


if __name__ == "__main__":

    env = simulationEnvironment({})

    state = env.observe_state()

    print(state)

    print(env.is_at_goal())

    for _ in range(3):

        for tt in range(3):
            env.execute([0, 0, 0])

            time.sleep(3)

        env.reset()

        # env.reset()
        #
        # env.observe_state()
        #
        # for tt in range(3):
        #   env.execute([0, -1, 0])

        # for tt in range(1000):
        #    env.execute([0, 0, -0.2])
