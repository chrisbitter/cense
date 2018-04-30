#Standard Imports
import array
import vrep
import time
vrep.simxFinish(-1) # just in case, close all opened connections
def Move_possible(pos):
    vrep.simxSynchronous(clientID, True)
    alpha = vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot_wait)
    vrep.simxSynchronousTrigger(clientID)
    print(dummyHandle)
    beta=vrep.simxSetObjectPosition(clientID,dummyHandle,-1,pos,vrep.simx_opmode_oneshot)
    time.sleep(5)

    vrep.simxSynchronous(clientID, False)
    gamma=vrep.simxStopSimulation(clientID,vrep.simx_opmode_oneshot_wait)
    [returnCode, signalValue] = vrep.simxGetStringSignal(clientID, 'Move', vrep.simx_opmode_oneshot_wait)
    text=signalValue.decode()
    print(text)
def Move_to_Pose(pos):

    vrep.simxSynchronous(clientID, True)
    alpha = vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot_wait)
    vrep.simxSynchronousTrigger(clientID)
    print(dummyHandle)
    beta=vrep.simxSetObjectPosition(clientID,dummyHandle,-1,pos,vrep.simx_opmode_oneshot)
    time.sleep(5)

    vrep.simxSynchronous(clientID, False)
    gamma=vrep.simxStopSimulation(clientID,vrep.simx_opmode_oneshot_wait)


clientID=vrep.simxStart('127.0.0.1',19997,True,True,5000,5)
if clientID!=-1:
    code,dummyHandle = vrep.simxGetObjectHandle(clientID,'UR5_target#', vrep.simx_opmode_oneshot_wait)
    print(dummyHandle)
    number_of_coordinates=int(input('Enter the number of coordinates to be entered'))

    for values in range(number_of_coordinates):
        x = float(input('Enter the value of x:'))
        y = float(input('Enter the value of y:'))
        z = float(input('Enter the value of z:'))
        pos=(x,y,z)
        print(pos)
        #Move_to_Pose(pos)
        Move_possible(pos)
