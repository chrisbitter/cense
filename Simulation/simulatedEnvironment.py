import array
import Simulation.vrep as vrep
import time
import numpy as np


class IllegalPoseException(Exception):
    def __init__(self, *args):
        super(IllegalPoseException, self).__init__(*args)


class simulationEnvironment(object):
    def __init__(self, config):
        self.ROTATION_MAX_ANGLE = np.pi / 2
        self.TRANSLATION_SIDEWAYS_MAX_DISTANCE = .03
        self.TRANSLATION_FORWARD_MAX_DISTANCE = .03
        self.ERROR_TRANSLATION = .001
        self.ERROR_ROTATION = 1 * np.pi / 180
        self.START_POSITION = [.31, -.2, 1.12]
        self.START_ROTATION = np.pi / 4

        self.CONSTRAINT_MIN = np.array([-.35, -.25, .7])
        self.CONSTRAINT_MAX = np.array([.35, -.15, 1.3])

        self.GOAL_X = -.25

        self.STATE_DIMENSIONS = (40, 40, 3)
        self.ACTIONS = 3
        self.PUNISHMENT_WIRE = -1
        self.PUNISHMENT_INSUFFICIENT_PROGRESS = -1

        vrep.simxFinish(-1)  # just in case, close all opened connections

        self.clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)

        if self.clientID != -1:
            _, self.targetHandle = vrep.simxGetObjectHandle(self.clientID, 'UR5_target#', vrep.simx_opmode_blocking)
            _, self.tipHandle = vrep.simxGetObjectHandle(self.clientID, 'UR5_tip', vrep.simx_opmode_blocking)
            _, self.collisionHandle = vrep.simxGetCollisionHandle(self.clientID, 'WireLoopCollision',
                                                                  vrep.simx_opmode_blocking)
            _, self.cameraHandle = vrep.simxGetObjectHandle(self.clientID, 'Camera', vrep.simx_opmode_blocking)

            vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_blocking)

            self.current_position = self.START_POSITION.copy()
            self.current_orientation = self.START_ROTATION

            self.reset()

    def __del__(self):
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_blocking)

    def reset(self, hard_reset=False):
        self.current_position = self.START_POSITION.copy()
        self.current_orientation = self.START_ROTATION

        self.move_to_pose(self.current_position, [-np.pi, self.current_orientation, np.pi])

    def is_at_goal(self):

        position, _ = self.get_pose()

        return self.GOAL_X > position[0]

    def update_current_start_pose(self):
        pass

    def move_to_pose(self, position, orientation):

        if position[0] < self.CONSTRAINT_MIN[0] or position[0] > self.CONSTRAINT_MAX[0] \
                or position[1] < self.CONSTRAINT_MIN[1] or position[1] > self.CONSTRAINT_MAX[1] \
                or position[2] < self.CONSTRAINT_MIN[2] or position[2] > self.CONSTRAINT_MAX[2]:
            print(position)
            raise IllegalPoseException

        touched_wire = False

        while True:

            while vrep.simxSetObjectOrientation(self.clientID, self.targetHandle, -1, orientation,
                                                vrep.simx_opmode_oneshot) != 0:
                pass

            while vrep.simxSetObjectPosition(self.clientID, self.targetHandle, -1, position,
                                             vrep.simx_opmode_oneshot) != 0:
                pass

            current_tip_position, current_tip_orientation = self.get_pose()
            current_target_position, current_target_orientation = self.get_target_pose()

            translation_deviation = np.sum(np.absolute(current_tip_position - current_target_position))
            rotation_deviation = np.sum(np.absolute(
                ((np.array(current_tip_orientation - current_target_orientation) + np.pi) % (2 * np.pi)) - np.pi))

            if translation_deviation < self.ERROR_TRANSLATION and rotation_deviation < self.ERROR_ROTATION:
                break

            _, collision = vrep.simxGetIntegerSignal(self.clientID, "Collision", vrep.simx_opmode_blocking)

            touched_wire |= collision

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

        self.current_position[0] -= action[0] * self.TRANSLATION_SIDEWAYS_MAX_DISTANCE * np.cos(
            self.current_orientation) \
                                    - action[1] * self.TRANSLATION_FORWARD_MAX_DISTANCE * np.sin(
            self.current_orientation)
        self.current_position[2] += action[0] * self.TRANSLATION_SIDEWAYS_MAX_DISTANCE * np.sin(
            self.current_orientation) \
                                    + action[1] * self.TRANSLATION_FORWARD_MAX_DISTANCE * np.cos(
            self.current_orientation)

        self.current_orientation += action[2] * self.ROTATION_MAX_ANGLE
        self.current_orientation = self.current_orientation % (2 * np.pi)

        new_orientation = [-np.pi, self.current_orientation, np.pi]

        # print(new_orientation[1])
        touched_wire = self.move_to_pose(self.current_position, new_orientation)

        # print(touched_wire)

        reward = action[0]

        if touched_wire:
            reward = -1
            self.reset()
            vrep.simxSetIntegerSignal(self.clientID, 'Collision', 0, vrep.simx_opmode_blocking)

        return self.observe_state(), reward, touched_wire

        # if touched_wire:
        #     print("Touched")
        # else:
        # #    self.reset()


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
