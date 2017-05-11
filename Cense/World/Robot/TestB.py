import RTDE_Controller_CENSE as rtde

print(rtde.current_position())

rtde.go_start_via_path()

rtde.go_camera()

#rtde.go_start_via_path()


rtde.disconnect_rtde()
