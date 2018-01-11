from Cense.Environment.Robot.rtdeController import RtdeController
from Cense.Environment.continuousEnvironment import ContinuousEnvironment as Environment
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches


controller = RtdeController()

try:
    pose, _ = controller.current_pose()
    controller.move_to_pose(pose)
except:
    pass

fig1 = plt.figure()

ax1 = fig1.add_subplot(121, aspect='equal')
ax1.add_patch(
    patches.Rectangle(
        (controller.CONSTRAINT_MIN[0], controller.CONSTRAINT_MIN[2]),  # (x,y)
        controller.CONSTRAINT_MAX[0] - controller.CONSTRAINT_MIN[0],  # width
        controller.CONSTRAINT_MAX[2] - controller.CONSTRAINT_MIN[2],  # height
        fill=False
    )
)


#ax1.invert_yaxis()
# ax1.relim()
# ax1.autoscale_view()

ax2 = fig1.add_subplot(122, aspect='equal')
ax2.add_patch(
    patches.Rectangle(
        (controller.CONSTRAINT_MIN[1], controller.CONSTRAINT_MIN[2]),  # (x,y)
        controller.CONSTRAINT_MAX[1] - controller.CONSTRAINT_MIN[1],  # width
        controller.CONSTRAINT_MAX[2] - controller.CONSTRAINT_MIN[2],  # height
        fill=False
    )
)

ax2.invert_xaxis()
#ax2.invert_yaxis()

ax1.plot(Environment.START_POSE[0], Environment.START_POSE[2], 'ro')
ax2.plot(Environment.START_POSE[1], Environment.START_POSE[2], 'ro')

ax1.axvline(Environment.GOAL_X)

position_xz_plot, = ax1.plot(controller.CONSTRAINT_MIN[0], controller.CONSTRAINT_MIN[2], 'bo')
position_yz_plot, = ax2.plot(controller.CONSTRAINT_MIN[1], controller.CONSTRAINT_MIN[2], 'bo')

while True:
    pose = controller.current_pose()

    pose = list(pose[0][:3])

    print("Pose:", pose)
    print("At Goal:", pose[0] < Environment.GOAL_X)

    position_xz_plot.set_xdata(pose[0])
    position_xz_plot.set_ydata(pose[2])

    position_yz_plot.set_xdata(pose[1])
    position_yz_plot.set_ydata(pose[2])

    plt.draw()
    plt.pause(.001)
