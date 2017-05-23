from Cense.World.Robot.rtde_controller import RTDE_Controller
import time

controller = RTDE_Controller()

if True:
    pose = controller.current_position()

    print("Pose:", pose)

    if pose:

        for _ in range(20):

            pose[0] += .05
            controller.move_to_pose(pose)

            pose[2] += .05
            controller.move_to_pose(pose)

            now = time.time()
            while time.time() - now < 5:
                pass

            pose[0] -= .05
            controller.move_to_pose(pose)

            pose[2] -= .05
            controller.move_to_pose(pose)

