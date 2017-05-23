import Cense.World.Robot.RTDE_Controller_CENSE as rtde
while True:
    print(rtde.Positions.all_positions)

    rtde.go_start_via_path()
    print ('Start Position')

    rtde.move_to_position([-0.10782, 0.3822, 0.5866, 0.0, -0.136, 1.593])
    print ('Position 1')
    rtde.move_to_position([-0.10782, 0.4822, 0.5866, 0.0, -0.136, 1.593])
    print ('Position 2')
    rtde.move_to_position([-0.10782, 0.4822, 0.4866, 0.0, -0.136, 1.593])
    print ('Position 3')
    rtde.move_to_position([-0.10782, 0.3822, 0.4866, 0.0, -0.136, 1.593])
    print ('Position 4')

    rtde.go_start_via_path()
    print ('Start Position')

    print(rtde.Positions.all_positions)

rtde.disconnect_rtde()
