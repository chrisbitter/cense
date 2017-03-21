import World.Simulated.worldParser as worldParser
import numpy as np
from World.world import World
import math


class SimulatedWorld(World):

    # Array containing the current state of the world
    __world = np.zeros([5, 5])
    # Current position of the tool center point
    tcp_pos = [0, 0]
    # Static variable determining how big a state snapshot should be
    __state_size = 5

    def __init__(self, world_path=None):
        if world_path is not None:
            # Parse world from given image
            self.__world = worldParser.create_wire_from_file(world_path)
        if world_path is None:
            self.__world = worldParser.create_wire_from_file()

        self.__create_init_state()

    #
    # Initiates the claws and the first goal
    #
    def __create_init_state(self):
        # Calculate the offset of the claws from the center point of a given state
        tool_offset = math.ceil(self.__state_size/2)
        print("offset is :", tool_offset)
        self.tcp_pos = (tool_offset, 0)

        # Calculate worlds center based on the assumption that the wire always starts in the middle of the image
        center = math.floor(self.__world.shape[0]/2)
        print("center is :", center)
        if center < self.__state_size:
            print("Warning: image is to small to properly position claws")

        # Calculate initial claw positions based on the worlds center and the previously calculated offset
        upper_claw_pos = ((center - tool_offset), 0)
        lower_claw_pos = ((center + tool_offset), 0)
        self._set_flag(upper_claw_pos, self.rotor_top)
        self._set_flag(lower_claw_pos, self.rotor_bot)

        # Starting position of line is always in the exact middle (Might change some day)
        lineposx = self.tcp_pos[0]
        lineposy = self.tcp_pos[1]

    #
    # Sets a flag while maintaining all other flags that have been set
    #
    def _add_flag(self, position, state):
        if not self._flag_is_set(position, state):
            self.__world[position[0], position[1]] += state

    #
    # Removes a flag if it is set
    #
    def _remove_flag(self, position, state):
        if self._flag_is_set(position, state):
            self.__world[position[0], position[1]] -= state

    #
    # Sets a flag while removing all other set flags
    #
    def _set_flag(self, position, state):
        self.__world[position[0], position[1]] = state

    #
    # Function that checks whether a flag is set in a given position
    #
    def _flag_is_set(self, position, state):
        status = self.__world[position[0], position[1]]
        # Bit shift both bytes until the flag is in the first bit
        while state != 1:
            state >>= 1
            status >>= 1

        # Check if Flag is set in the status
        if status % 2 != 0:
            return True
        else:
            return False

    #
    # returns coordinates of a new goal based on the coordinates of the old goal
    #
    def find_new_goal(self, old_goal):
        position = np.array(old_goal)
        goal_column = old_goal[0] + self.__state_size
        path_from_old_goal = []
        # Positions relative to the old goal where the wire might continue
        positions_to_search = np.array([[0, 1], [0, -1], [1, 1], [1, 0], [1, -1]])
        while position[0] != goal_column:
            positions_found = []
            for i in positions_to_search:
                if i not in path_from_old_goal and self._flag_is_set(position+i, np.sum([position, i])):
                    positions_found.append(i)
            if len(positions_found) == 1:
                path_from_old_goal.append(position)
                position = np.sum(positions_found[0], position)
            else:
                # TODO: finish this, line 178 in HW_6act_lookup_4
                pass

    #
    # Returns the state with coordinates describing the upper left
    #
    def get_state(self, coordinates):
        return self.__world[coordinates[0]:coordinates[0] + self.__state_size, coordinates[1]:coordinates[1] + self.__state_size]

    #
    # runs a few tests
    #
    def test(self):
        print(self.__world.shape)
        # print(self.__world[0, 0])
        print(self.get_state([math.ceil(self.__world.shape[0]/2-self.__state_size/2), 0]))

    #
    # Moves the claws one unit to the right
    #
    def __move_right(self):
        pass

    def move_left(self):
        pass

    def move_right(self):
        pass

    def move_up(self):
        pass

    def move_down(self):
        pass

    def turn_clockwise(self):
        pass

    def turn_counter_clockwise(self):
        pass

if __name__ == '__main__':
    world = SimulatedWorld()
    world.test()
