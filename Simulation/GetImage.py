from PIL import Image
import array
import Simulation.vrep as vrep
import numpy as np
import matplotlib.pyplot as plt
vrep.simxFinish(-1) # just in case, close all opened connections

clientID=vrep.simxStart('127.0.0.1',19997,True,True,5000,5) # Connect to V-REP
if clientID!=-1:
    #x = vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot)
    res, v0 = vrep.simxGetObjectHandle(clientID, 'Vision_sensor', vrep.simx_opmode_oneshot_wait)
    print(v0, res)
    res, resolution, image = vrep.simxGetVisionSensorImage(clientID, v0, 0, vrep.simx_opmode_oneshot_wait)

    print(resolution)
    print(image)


    img = np.array(image, dtype=bytes).reshape(tuple(resolution) + (3,)).astype(np.ubyte)

    print(img)

    print(img.min(), img.max())

    plt.imshow(img)



    plt.show()

    #image_byte_array = array.array('b', image)
    #im = Image.frombuffer("RGB", (256, 144), image_byte_array, "raw", "RGB", 0, 1)
    #im.show()
else:
    print('Error')
# image obtained as a Image object. Use it according to need.