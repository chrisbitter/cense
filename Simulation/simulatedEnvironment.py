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
        self.START_POSITION = [.2, -.6, 1.07]

        vrep.simxFinish(-1)  # just in case, close all opened connections

        self.clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)

        if self.clientID != -1:
            _, self.targetHandle = vrep.simxGetObjectHandle(self.clientID, 'UR5_target#', vrep.simx_opmode_blocking)
            _, self.tipHandle = vrep.simxGetObjectHandle(self.clientID, 'UR5_tip', vrep.simx_opmode_blocking)

            vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_blocking)

            current_position, current_orientation = self.get_target_pose()

            self.current_position = self.START_POSITION
            self.current_orientation = 0

    def __del__(self):
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_blocking)

    def reset(self):
        vrep.simxStopSimulation(self.clientID, vrep.simx_opmode_blocking)
        vrep.simxStartSimulation(self.clientID, vrep.simx_opmode_blocking)

    def move_to_pose(self, position, orientation):

        while True:

            while vrep.simxSetObjectOrientation(self.clientID, self.targetHandle, -1, orientation,
                                                vrep.simx_opmode_blocking) != 0:
                pass

            while vrep.simxSetObjectPosition(self.clientID, self.targetHandle, -1, position,
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
        print('entzer')
        
        #print(current_position)

        self.current_position[0] -= action[0] * self.TRANSLATION_SIDEWAYS_MAX_DISTANCE * np.cos(self.current_orientation) \
                           - action[1] * self.TRANSLATION_FORWARD_MAX_DISTANCE * np.sin(self.current_orientation)
        self.current_position[2] += action[0] * self.TRANSLATION_SIDEWAYS_MAX_DISTANCE * np.sin(self.current_orientation) \
                           + action[1] * self.TRANSLATION_FORWARD_MAX_DISTANCE * np.cos(self.current_orientation)

        self.current_orientation += action[2] * self.ROTATION_MAX_ANGLE
        self.current_orientation = self.current_orientation % (2*np.pi)

        new_orientation = [-np.pi, self.current_orientation, np.pi]

        #print(new_position[1])
        self.move_to_pose(self.current_position, new_orientation)


if __name__ == "__main__":

    env = simulationEnvironment()


    for tt in range(10):
        env.execute([0, 1, 0])
