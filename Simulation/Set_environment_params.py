
import vrep
import sys
import ctypes
import time
import string
#Function to set environment variables(Color and path of texture file)
def Set_environment_params(red, green, blue, loc):
    vrep.simxSetFloatSignal(clientID, 'Red', red, vrep.simx_opmode_oneshot)
    vrep.simxSetFloatSignal(clientID, 'Green',green, vrep.simx_opmode_oneshot)
    vrep.simxSetFloatSignal(clientID, 'Blue', blue, vrep.simx_opmode_oneshot)
    str_bytes = str.encode(loc)
    print(str_bytes)
    raw_ubytes = (ctypes.c_ubyte * len(str_bytes)).from_buffer_copy(str_bytes)
    a = vrep.simxSetStringSignal(clientID, b'Path', raw_ubytes, vrep.simx_opmode_oneshot)
    print(' Color and Texture Succesfully changed')

#Code to start the API connection
print ('Program started')
vrep.simxFinish(-1) # just in case, close all opened connections
clientID=vrep.simxStart('127.0.0.1',19997,True,True,5000,5) # Connect to V-REP

if clientID!=-1:
    x = vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot_wait)
    print ('Connected to remote API server')
#Setting the user inputs here
    r = float(input('Enter the red component of the color between values 0-1:'))
    g = float(input('Enter the green component of the color between values 0-1:'))
    b = float(input('Enter the blue component of the color between values 0-1:'))
    path=input('Enter the path of the location')
#calling the function
    Set_environment_params(r, g, b, path)

    x = vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot_wait)

else:
    print ('Remote API function call returned with error code: ')
    sys.exit('Could not connect')
print('End of simulation')


