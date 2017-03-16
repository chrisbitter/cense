import numpy as np
import World.worldParser as worldParser
import Decider.action as action


#
# World Object representing the world in it's current state.
# performs Actions defined in the Action package
#
class World:
    # Array containing the current state of the world
    __worldState = np.zeros([5, 5, 4])
    # Current position of the tool center point
    tcp_pos_x = 0
    tcp_pos_y = 0

    def __init__(self, world_path=None):
        if world_path is not None:
            # Parse world from given image
            self.__worldState = worldParser.create_wire_from_file(world_path)
        if world_path is None:
            self.__worldState = worldParser.create_wire_from_file()

    def perform_action(self, action_to_perform):
        if not isinstance(action_to_perform, action.Action):
            print("Error: Action is not in the allowed set of Actions")
            return

        if action_to_perform == action.Action.LEFT:
            print("perform left")
        elif action_to_perform == action.Action.RIGHT:
            print("perform right")
        elif action_to_perform == action.Action.UP:
            print("perform up")
        elif action_to_perform == action.Action.DOWN:
            print("perform down")
        elif action_to_perform == action.Action.ROTATE_CLOCKWISE:
            print("rotate clockwise")
        elif action_to_perform == action.Action.ROTATE_COUNTER_CLOCKWISE:
            print("rotate counterclockwise")
