from Cense.World.Robot.rtde_controller import RTDE_Controller, IllegalPoseException
import time
import numpy as np
import math


class TestWorld:
    controller = None

    def __init__(self):
        self.controller = RTDE_Controller()

        pose = [self.controller.X_MIN, self.controller.Y_MIN, self.controller.Z_MIN, 0, 0, 0]

        self.controller.move_to_pose(pose)

    def test_movement(self):
        pose = self.controller.current_pose()

        print("Pose:", pose)

        if pose:
            for _ in range(20):
                pose[0] += .05
                self.controller.move_to_pose(pose)

                pose[2] += .05
                self.controller.move_to_pose(pose)

                pose[0] -= .05
                self.controller.move_to_pose(pose)

                pose[2] -= .05
                self.controller.move_to_pose(pose)

    def test_relative_movement(self):
        pose = self.controller.current_pose()

        move_x = .005
        move_y = 0

        if pose:
            for _ in range(20):
                print("Pose:", pose)

                pose[4] = (np.random.random() * 2 - 1) * math.pi

                pose[0] = math.cos(pose[4]) * move_x - math.sin(
                    pose[4]) * move_y
                pose[2] = math.sin(pose[4]) * move_x - math.cos(
                    pose[4]) * move_y

                print("Relative Movement:", move_x, move_y)
                print("New Pose:", pose)

                input()

                self.controller.move_to_pose(pose)

    def test_contraint_box(self):
        pose = self.controller.current_pose()

        if pose:
            # A
            pose = [self.controller.X_MIN, self.controller.Y_MIN, self.controller.Z_MIN, 0, 0, 0]
            self.controller.move_to_pose(pose)

            # A-D
            pose = [self.controller.X_MAX, self.controller.Y_MIN, self.controller.Z_MIN, 0, 0, 0]
            self.controller.move_to_pose(pose)
            pose = [self.controller.X_MIN, self.controller.Y_MIN, self.controller.Z_MIN, 0, 0, 0]
            self.controller.move_to_pose(pose)

            # A-A'-A
            pose = [self.controller.X_MIN, self.controller.Y_MAX, self.controller.Z_MIN, 0, 0, 0]
            self.controller.move_to_pose(pose)
            pose = [self.controller.X_MIN, self.controller.Y_MIN, self.controller.Z_MIN, 0, 0, 0]
            self.controller.move_to_pose(pose)

            # A-B-A
            pose = [self.controller.X_MIN, self.controller.Y_MIN, self.controller.Z_MAX, 0, 0, 0]
            self.controller.move_to_pose(pose)
            pose = [self.controller.X_MIN, self.controller.Y_MIN, self.controller.Z_MIN, 0, 0, 0]
            self.controller.move_to_pose(pose)

            # A-C'-A
            pose = [self.controller.X_MAX, self.controller.Y_MAX, self.controller.Z_MAX, 0, 0, 0]
            self.controller.move_to_pose(pose)
            pose = [self.controller.X_MIN, self.controller.Y_MIN, self.controller.Z_MIN, 0, 0, 0]
            self.controller.move_to_pose(pose)

    def test_constraint_box_violation(self):
        pose = self.controller.current_pose()

        if pose:
            # A
            pose = [self.controller.X_MIN, self.controller.Y_MIN, self.controller.Z_MIN, 0, 0, 0]
            self.controller.move_to_pose(pose)

            try:
                while True:
                    pose[0] += .01
                    self.controller.move_to_pose(pose)

            except IllegalPoseException:
                print("Controller threw IllegalPoseException!")


if __name__ == "__main__":
    testWorld = TestWorld()

    # testWorld.test_movement()

    # testWorld.test_relative_movement()

    # testWorld.test_contraint_box()

    testWorld.test_constraint_box_violation()
