from Cense.World.Robot.rtde_controller import RTDE_Controller
import time

timeout = 3

controller = RTDE_Controller()

while True:
    pose = controller.current_pose()

    print("Pose:", pose)

    now = time.time()
    while time.time() - now < timeout:
        pass