# Standard Imports
import array
import Simulation.vrep as vrep
import time
import numpy as np

class simulationEnvironment(object):
    def __init__(self):
        self.ROTATION_MAX_ANGLE = np.pi / 2
        self.TRANSLATION_SIDEWAYS_MAX_DISTANCE = .03
        self.TRANSLATION_FORWARD_MAX_DISTANCE = .03
        self.ERROR_TRANSLATION = .001
        self.ERROR_ROTATION = 1 * np.pi / 180

        vrep.simxFinish(-1)  # just in case, close all opened connections

        self.clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)

        if self.clientID != -1:
            _, self.targetHandle = vrep.simxGetObjectHandle(self.clientID, 'UR5_target#', vrep.simx_opmode_blocking)
            _, self.tipHandle = vrep.simxGetObjectHandle(self.clientID, 'UR5_tip', vrep.simx_opmode_blocking)

            vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_blocking)

    def __del__(self):
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_blocking)

    def reset(self):
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_blocking)
        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_blocking)

    def move_to_pose(self, position, orientation):

        while True:

            while vrep.simxSetObjectPosition(self.clientID, self.targetHandle, -1, position,
                                             vrep.simx_opmode_blocking) != 0:
                pass

            while vrep.simxSetObjectOrientation(self.clientID, self.targetHandle, -1, orientation,
                                                vrep.simx_opmode_blocking) != 0:
                pass

            current_tip_position, current_tip_orientation = self.get_pose()
            current_target_position, current_target_orientation = self.get_target_pose()

            translation_deviation = np.sum(np.absolute(current_tip_position - current_target_position))
            rotation_deviation = np.sum(np.absolute(
                ((np.array(current_tip_orientation - current_target_orientation) + np.pi) % (2 * np.pi)) - np.pi))

            if translation_deviation < self.ERROR_TRANSLATION and rotation_deviation < self.ERROR_ROTATION:
                break

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

    def execute(self, action):

        current_position, current_orientation = self.get_target_pose()

        new_position = current_position.copy()
        new_orientation = current_orientation.copy()

        print(current_orientation)

        new_position[0] -= action[0] * self.TRANSLATION_SIDEWAYS_MAX_DISTANCE * np.cos(current_orientation[1]) \
                           - action[1] * self.TRANSLATION_FORWARD_MAX_DISTANCE * np.sin(current_orientation[1])
        new_position[2] += action[0] * self.TRANSLATION_SIDEWAYS_MAX_DISTANCE * np.sin(current_orientation[1]) \
                           + action[1] * self.TRANSLATION_FORWARD_MAX_DISTANCE * np.cos(current_orientation[1])

        new_orientation[1] += action[2] * self.ROTATION_MAX_ANGLE

        print(new_orientation)

        self.move_to_pose(new_position, new_orientation)


if __name__ == "__main__":

    env = simulationEnvironment()

    for tt in range(1000):
        env.execute([0, 0, .5])
