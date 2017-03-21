import World.Simulated.worldParser as worldParser
import numpy as np
from World.world import World
import math
from Decider.action import Action


class SimulatedWorld(World):

    # Array containing the current state of the world
    __world = np.zeros([5, 5])
    # Current position of the tool center point
    tcp_pos = [0, 0]
    # Static variable determining how big a state snapshot should be
    __state_size = 5

    def __init__(self, world_path=None):
        super().__init__()
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
        tool_offset = math.floor(self.__state_size/2)

        # Calculate worlds center based on the assumption that the wire always starts in the middle of the image
        center = math.floor(self.__world.shape[0]/2)
        self.tcp_pos = (center, 0)
        if center < self.__state_size:
            print("Warning: image is to small to properly position claws")

        # Calculate initial claw positions based on the worlds center and the previously calculated offset
        upper_claw_pos = ((center - tool_offset), 0)
        lower_claw_pos = ((center + tool_offset), 0)
        self._set_flag(upper_claw_pos, self.rotor_top)
        self._set_flag(lower_claw_pos, self.rotor_bot)

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
    def find_new_goal(self, previous_action):
        # Make a list of all wire positions around the old goal
        list_of_wires = []
        state_offset = [self.tcp_pos[0]-math.floor(self.__state_size/2), self.tcp_pos[1] + math.floor(self.__state_size/2)]
        for i in range(self.__state_size):
            for j in range(self.__state_size):
                if self._flag_is_set([state_offset[1]+j, state_offset[0]+i], self.wire):
                    list_of_wires.append([i, j])

        # Clean list
        if previous_action == Action.UP:
            list_of_wires = list_of_wires[list_of_wires[:, 0] <= 2]
            # TODO: finish implementation

        distance_list = []
        for wire in list_of_wires:
            x_dist = np.abs(2 - wire[0])
            y_dist = np.abs(2 - wire[1])
            dist = x_dist + y_dist
            distance_list.append(dist)

        goal_index = np.argmax(distance_list)
        goal_relative = list_of_wires[goal_index]
        goal = np.sum(goal_relative, state_offset)
        return goal

    #
    # Returns the state with coordinates describing the upper left
    #
    def get_state(self, coordinates):
        # Create variables representing the upper left corner of the state
        state_start_x = coordinates[0]
        state_start_y = coordinates[1]

        # Create variables representing the size of the world
        world_size_x = self.__world.shape[1]
        world_size_y = self.__world.shape[0]

        # Create variables representing the size of a state
        x_size = self.__state_size
        y_size = self.__state_size

        # Claws always have to be in the middle of the state, so check if there is enough space to the left to do this
        init_state = False
        if state_start_x <= math.floor(self.__state_size/2):
            # Not enough space, extend with out of world
            init_state = True

        if init_state:
            x_size = math.ceil(self.__state_size/2) + state_start_x

        # Make sure the snapshot is within boundaries
        if x_size+state_start_x > world_size_x:
            # x_offset is too close to the border, would try to copy nonexistent indices, adjust size of copy
            x_size = world_size_x - state_start_x
        if y_size+state_start_y > world_size_y:
            # y_offset is too close to the border, would try to copy nonexistent indices, adjust size of copy
            y_size = world_size_y - state_start_y

        # Do the copy of the selected area
        state = self.__world[state_start_y:state_start_y + y_size, state_start_x:state_start_x + x_size]

        # Fill the resized area with out of world Flag
        if x_size < self.__state_size:
            if not init_state:
                # X went out of world, extend towards the right
                out_of_world_stack = np.zeros([state.shape[0], self.__state_size - x_size], dtype=np.uint8)
                out_of_world_stack.fill(self.out_of_world)
                state = np.hstack((state, out_of_world_stack))
            else:
                # X is smaller because we are in the initial state where we have to extend to the left
                out_of_world_stack = np.zeros([state.shape[0], self.__state_size - x_size], dtype=np.uint8)
                out_of_world_stack.fill(self.out_of_world)
                state = np.hstack((out_of_world_stack, state))
        if y_size < self.__state_size:
            # Y went out of world, extend towards the bottom
            out_of_world_stack = np.zeros([self.__state_size-y_size, state.shape[1]], dtype=np.uint8)
            out_of_world_stack.fill(self.out_of_world)
            state = np.vstack((state, out_of_world_stack))
        return state

    #
    # runs a few tests
    #
    def test(self):
        print(self.__world.shape)
        # Check if in world states work
        # in_world_coords = [math.ceil(self.__world.shape[0]/2-self.__state_size/2), 0]
        # print("Check if in world works with coords: ", in_world_coords)
        # print(self.get_state(in_world_coords))

        # Check if out of world x works
        # out_of_world_x_coords = [self.__world.shape[1]-self.__state_size + 1, self.__world.shape[0]-self.__state_size]
        # print("Check if x works with coords: ", out_of_world_x_coords)
        # print(self.get_state(out_of_world_x_coords))

        # Check if out of world y works
        # out_of_world_y_coords = [self.__world.shape[1]-self.__state_size, self.__world.shape[0]-self.__state_size + 1]
        # print("Check if y works with coords: ", out_of_world_y_coords)
        # print(self.get_state(out_of_world_y_coords))

        # Check if out of world x and y works
        # print("Check if x and y works")
        # out_of_world_y_coords = [self.__world.shape[1] - self.__state_size + 1,
        #                         self.__world.shape[0] - self.__state_size + 1]
        # print("Check if x and y works with coords: ", out_of_world_y_coords)
        # print(self.get_state(out_of_world_y_coords))

        # Check if init state works
        init_state_coords = [0, math.floor(self.__world.shape[0]/2)]
        print("Check if init state works with coords: ", init_state_coords)
        print(self.get_state(init_state_coords))

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
