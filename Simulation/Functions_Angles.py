# Standard Imports
import array
import Simulation.vrep as vrep
import time
import numpy as np

vrep.simxFinish(-1)  # just in case, close all opened connections


def Move_possible(pos, angles):
    # print(dummyHandle)
    vrep.simxSetObjectPosition(clientID, dummyHandle, -1, pos, vrep.simx_opmode_oneshot)
    # vrep.simxSynchronousTrigger(clientID)
    vrep.simxSetObjectOrientation(clientID, dummyHandle, -1, angles, vrep.simx_opmode_oneshot)
    # vrep.simxCallScriptFunction(clientID,'UR5#0',vrep.sim_scripttype_childscript,'Move_Possible',[],[],[],bytearray([]),vrep.simx_opmode_oneshot_wait)
    returnCode, signalValue = vrep.simxGetStringSignal(clientID, 'Move', vrep.simx_opmode_oneshot_wait)
    text = signalValue.decode()
    print(text)


def Move_to_Pose(pos, angles):
    vrep.simxSynchronous(clientID, True)
    alpha = vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot_wait)
    vrep.simxSynchronousTrigger(clientID)
    print(dummyHandle)
    beta = vrep.simxSetObjectPosition(clientID, dummyHandle, -1, pos, vrep.simx_opmode_oneshot)
    nu = vrep.simxSetObjectOrientation(clientID, dummyHandle, -1, angles, vrep.simx_opmode_oneshot)
    vrep.simxCallScriptFunction(clientID, 'UR5#0', vrep.sim_scripttype_childscript, 'Move_to_Pose', [], [], [],
                                bytearray([]), vrep.simx_opmode_blocking)
    time.sleep(5)

    vrep.simxSynchronous(clientID, False)
    gamma = vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot_wait)


if __name__ == "__main__":
    clientID = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5)
    if clientID != -1:
        code, dummyHandle = vrep.simxGetObjectHandle(clientID, 'UR5_target#', vrep.simx_opmode_oneshot_wait)
        print(dummyHandle)
        # number_of_coordinates=int(input('Enter the number of coordinates to be entered'))
        number_of_coordinates = 1

        vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot)

        # vrep.simxSynchronousTrigger(clientID)

        for tt in range(10000):
            for values in range(number_of_coordinates):
                x = .3
                y = .3 + .5 * (np.sin(tt / 500) - .5)
                z = 1

                a = 0
                b = 0
                g = 0

                pos = (x, y, z)
                angles = (a, b, g)
                print(pos)
                # Move_to_Pose(pos,angles)
                # vrep.simxSynchronous(clientID, True)
                Move_possible(pos, angles)
                # vrep.simxSynchronous(clientID, False)


        vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot)