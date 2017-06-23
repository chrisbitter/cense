from Cense.World.Robot.rtdeController import RtdeController
import time

timeout = 3

controller = RtdeController(print)

while True:
    pose = controller.current_pose()

    print("Pose:", pose)

    now = time.time()
    while time.time() - now < timeout:
        pass
