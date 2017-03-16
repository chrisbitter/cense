import World.Simulated.worldParser as worldParser
import numpy as np
import World.world


class SimulatedWorld(World):

    # Array containing the current state of the world
    __world = np.zeros([5, 5])
    # Current position of the tool center point
    tcp_pos = 0, 0

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

        # Determine size of the state
        state_size = 9

        # Calculate the offset of the claws from the center point of a given state
        tool_offset = (state_size-1)/2

        # Calculate worlds center based on the assumption that the cable always starts in the middle of the image
        center = (self.__world.shape[0]-1)/2
        if center < 3:
            print("Warning: image is to small to properly position claws")

        # Calculate initial claw positions based on the worlds center and the previously calculated offset
        upper_claw_pos = (center - tool_offset), 0
        lower_claw_pos = (center + tool_offset), 0
        self._set_flag(upper_claw_pos, self.rotor_top)
        self._set_flag(lower_claw_pos, self.rotor_bot)

    def __move_right(self):
        pass

    def _add_flag(self, position, state):
        if not self._flag_is_set(position, state):
            self.__world[position] += state

    def _remove_flag(self, position, state):
        if self._flag_is_set(position, state):
            self.__world[position] -= state

    def _set_flag(self, position, state):
        self.__world[position] = state

    #
    # Function that checks whether a flag is set in a given position
    #
    def _flag_is_set(self, position, state):
        status = self.__world[position]
        # Bit shift both bytes until the flag is in the first bit
        while state != 1:
            state >>= 1
            status >>= 1

        # Check if Flag is set in the status
        if status % 2 != 0:
            return True
        else:
            return False
