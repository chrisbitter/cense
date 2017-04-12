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
    # Possible positions for the rotators, in order of clockwise rotation
    # Currently hardcoded for a state size of 5 TODO: create these dynamically
    positions_R_top = np.array([(0, 2), (1, 3), (2, 4), (3, 3), (4, 2), (3, 1), (2, 0), (1, 1)])
    positions_R_bot = np.array([(4, 2), (3, 1), (2, 0), (1, 1), (0, 2), (1, 3), (2, 4), (3, 3)])
    # Coordinates of the final goal the agent should reach
    last_goal = (0, 0)

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
        # Calculate final goal
        for i in range(self.__world.shape[1] - 1):
            if self._flag_is_set((self.__world.shape[0] - 1, i), self.wire):
                self.last_goal = (self.__world.shape[0] - 1, i)
        # Calculate the offset of the claws from the center point of a given state
        tool_offset = math.floor(self.__state_size / 2)

        # Calculate worlds center based on the assumption that the wire always starts in the middle of the image
        center = math.floor(self.__world.shape[1] / 2)
        self.tcp_pos = (0, center)
        if center < math.floor(self.__state_size / 2):
            print("Warning: image is to small to properly position claws")

        # Calculate initial claw positions based on the worlds center and the previously calculated offset
        upper_claw_pos = (0, (center - tool_offset))
        lower_claw_pos = (0, (center + tool_offset))
        self._set_flag(upper_claw_pos, self.rotor_top)
        self._set_flag(lower_claw_pos, self.rotor_bot)

        # Set initial goal
        goal_pos = (tool_offset, center)
        self._add_flag(goal_pos, self.goal)

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
        # Calculate upper left of the current state based on the tool center point position
        state_offset = [self.tcp_pos[0] - math.floor(self.__state_size / 2), self.tcp_pos[1] +
                        math.floor(self.__state_size / 2)]
        for i in range(self.__state_size):
            for j in range(self.__state_size):
                if self._flag_is_set([state_offset[1] + j, state_offset[0] + i], self.wire):
                    list_of_wires.append([i, j])

        # Clean list
        if previous_action == Action.UP:
            list_of_wires = list_of_wires[list_of_wires[:, 0] <= 2]
        if previous_action == Action.DOWN:
            list_of_wires = list_of_wires[list_of_wires[:, 0] >= 2]
        if previous_action == Action.LEFT:
            list_of_wires = list_of_wires[list_of_wires[:, 1] <= 2]
        if previous_action == Action.RIGHT:
            list_of_wires = list_of_wires[list_of_wires[:, 1] >= 2]

        # Calculate distance from original point to each potential goal
        distance_list = []
        for wire in list_of_wires:
            x_dist = np.abs(2 - wire[0])
            y_dist = np.abs(2 - wire[1])
            dist = x_dist + y_dist
            distance_list.append(dist)

        # Pick the potential goal with the highest distance
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
        world_size_x = self.__world.shape[0]
        world_size_y = self.__world.shape[1]

        # Create variables representing the size of a state
        x_size = self.__state_size
        y_size = self.__state_size

        # Claws always have to be in the middle of the state, so check if there is enough space to the left to do this
        init_state = False
        if state_start_x <= math.floor(self.__state_size / 2):
            # Not enough space, extend with out of world
            init_state = True
            x_size = math.ceil(self.__state_size / 2) + state_start_x

        # Make sure the snapshot is within boundaries
        if x_size + state_start_x > world_size_x:
            # x_offset is too close to the border, would try to copy nonexistent indices, adjust size of copy
            x_size = world_size_x - state_start_x
        if y_size + state_start_y > world_size_y:
            # y_offset is too close to the border, would try to copy nonexistent indices, adjust size of copy
            y_size = world_size_y - state_start_y

        # Do the copy of the selected area
        state = self.__world[state_start_x:state_start_x + x_size, state_start_y:state_start_y + y_size]

        # Fill the resized area with out of world Flag
        if x_size < self.__state_size:
            # Create filler Array
            out_of_world_stack = np.zeros([self.__state_size - x_size, state.shape[1]], dtype=np.uint8)
            out_of_world_stack.fill(self.out_of_world)
            if not init_state:
                # X went out of world, extend towards the right
                state = np.vstack((state, out_of_world_stack))
            else:
                # X is smaller because we are in the initial state where we have to extend to the left
                state = np.vstack((out_of_world_stack, state))
        if y_size < self.__state_size:
            # Y went out of world, extend towards the bottom
            out_of_world_stack = np.zeros([state.shape[0], self.__state_size - y_size], dtype=np.uint8)
            out_of_world_stack.fill(self.out_of_world)
            state = np.hstack((state, out_of_world_stack))
        return state

    #
    # Returns the state around the tcp
    #
    def get_state_by_tcp(self):
        state_position = self.get_current_state_coordinates()
        return self.get_state(state_position)

    #
    # Calculates the coordinates of the upper left of the current state based on the tcp
    #
    def get_current_state_coordinates(self):
        return [self.tcp_pos[0] - math.floor(self.__state_size / 2), self.tcp_pos[1] +
                math.floor(self.__state_size / 2)]

    #
    # Moves the tcp one unit to it's right and adjusts the positions of the claws
    #
    def move_right(self):
        claw_bot, claw_top = self.find_claw_positions()

        # Calculate the future positions of the claws
        claw_bot_next = (claw_bot[0] + 1, claw_bot[1])
        claw_top_next = (claw_top[0] + 1, claw_top[1])

        # Move the claws
        self.__move_claws(claw_bot, claw_bot_next, claw_top, claw_top_next)

        # Align tool center point
        self.tcp_pos[0] += 1

    #
    # Moves the tcp one unit to it's left and adjusts the positions of the claws
    #
    def move_left(self):
        claw_bot, claw_top = self.find_claw_positions()

        # Calculate the future positions of the claws
        claw_bot_next = (claw_bot[0] - 1, claw_bot[1])
        claw_top_next = (claw_top[0] - 1, claw_top[1])

        # Move the claws
        self.__move_claws(claw_bot, claw_bot_next, claw_top, claw_top_next)

        # Align tool center point
        self.tcp_pos[0] -= 1

    #
    # Moves the tcp one unit upwards and adjusts the positions of the claws
    #
    def move_up(self):
        claw_bot, claw_top = self.find_claw_positions()

        # Calculate the future positions of the claws
        claw_bot_next = (claw_bot[0], claw_bot[1] + 1)
        claw_top_next = (claw_top[0], claw_top[1] + 1)

        # Move the claws
        self.__move_claws(claw_bot, claw_bot_next, claw_top, claw_top_next)

        # Align tool center point
        self.tcp_pos[1] += 1

    #
    # Moves the tcp one unit downwards and adjusts the positions of the claws
    #
    def move_down(self):
        claw_bot, claw_top = self.find_claw_positions()

        # Calculate the future positions of the claws
        claw_bot_next = (claw_bot[0], claw_bot[1] - 1)
        claw_top_next = (claw_top[0], claw_top[1] - 1)

        # Move the claws
        self.__move_claws(claw_bot, claw_bot_next, claw_top, claw_top_next)

        # Align tool center point
        self.tcp_pos[1] -= 1

    #
    # Turns the tcp one unit clockwise
    # Currently only implemented for a state size of 5
    #
    def turn_clockwise(self):
        # Get the relative claw positions
        claw_bot, claw_top = self.find_claw_positions()
        state_offset = self.get_current_state_coordinates()
        claw_bot[0] -= state_offset[0]
        claw_bot[1] -= state_offset[1]
        claw_top[0] -= state_offset[0]
        claw_top[1] -= state_offset[1]

        # Calculate the future positions of the claws
        # Find position in the position list for top claw
        for idx, pos in enumerate(self.positions_R_top):
            if np.array_equal(pos, claw_top):
                r_top_idx = idx
        # Go to the next element and check for overflow
        if r_top_idx == self.positions_R_top.shape[0] - 1:
            claw_top_next = tuple(self.positions_R_top[0])
        else:
            claw_top_next = tuple(self.positions_R_top[r_top_idx + 1])

        # Find position in the position list for bottom claw
        for idx, pos in enumerate(self.positions_R_bot):
            if np.array_equal(pos, claw_bot):
                r_bot_idx = idx
        # Go to the next element and check for overflow
        if r_bot_idx == self.positions_R_top.shape[0] - 1:
            claw_bot_next = tuple(self.positions_R_bot[0])
        else:
            claw_bot_next = tuple(self.positions_R_bot[r_bot_idx + 1])

        # Move the claws
        self.__move_claws(claw_bot, claw_bot_next, claw_top, claw_top_next)
        # No need to adjust tcp, because it stays the same

    #
    # Turns the tcp one unit counter clockwise
    # Currently only implemented for a state size of 5
    #
    def turn_counter_clockwise(self):
        # Get the relative claw positions
        claw_bot, claw_top = self.find_claw_positions()
        state_offset = self.get_current_state_coordinates()
        claw_bot[0] -= state_offset[0]
        claw_bot[1] -= state_offset[1]
        claw_top[0] -= state_offset[0]
        claw_top[1] -= state_offset[1]

        # Calculate the future positions of the claws
        # Find position in the position list for top claw
        for idx, pos in enumerate(self.positions_R_top):
            if np.array_equal(pos, claw_top):
                r_top_idx = idx
        # Go to the next element and check for overflow
        if r_top_idx == 0:
            claw_top_next = tuple(self.positions_R_top[self.positions_R_top.shape[0] - 1])
        else:
            claw_top_next = tuple(self.positions_R_top[r_top_idx - 1])

        # Find position in the position list for bottom claw
        for idx, pos in enumerate(self.positions_R_bot):
            if np.array_equal(pos, claw_bot):
                r_bot_idx = idx
        # Go to the next element and check for overflow
        if r_bot_idx == 0:
            claw_bot_next = tuple(self.positions_R_bot[self.positions_R_top.shape[0] - 1])
        else:
            claw_bot_next = tuple(self.positions_R_bot[r_bot_idx - 1])

        # Move the claws
        self.__move_claws(claw_bot, claw_bot_next, claw_top, claw_top_next)
        # No need to adjust tcp, because it stays the same

    #
    # Returns the position of the bottom claw and the top claw
    #
    def find_claw_positions(self):
        # Calculate upper left of the current state based on the tool center point position
        state_offset = [self.tcp_pos[0] - math.floor(self.__state_size / 2), self.tcp_pos[1] +
                        math.floor(self.__state_size / 2)]
        # Search for the lower and the upper claw
        rotor_top, rotor_bot = (0, 0)
        for i in range(self.__state_size):
            for j in range(self.__state_size):
                if self._flag_is_set([state_offset[0] + i, state_offset[1] + j], self.rotor_bot):
                    rotor_bot = [i, j]
                elif self._flag_is_set([state_offset[0] + i, state_offset[1] + j], self.rotor_top):
                    rotor_top = [i, j]
        return rotor_bot, rotor_top

    #
    # Moves the claws from the old positions to the new ones if it is safe to do so, otherwise raises an Exception
    #
    def __move_claws(self, claw_bot, claw_bot_next, claw_top, claw_top_next):
        # Check if new position would be out of map
        next_positions = [claw_top_next, claw_bot_next]

        # Upper boundaries
        for next_position in next_positions:
            if self.__world.shape[0] < next_position[0]:
                raise ClawOutOfWorldException
            if self.__world.shape[1] < next_position[1]:
                raise ClawOutOfWorldException

        # Lower boundaries
        for next_position in next_positions:
            for i in next_position:
                if i < 0:
                    raise ClawOutOfWorldException

        # Check if new claws would collide with the wire
        for next_position in next_positions:
            if self._flag_is_set(next_position, self.wire):
                raise WireCollisionException

        # Remove old claw flags
        self._remove_flag(claw_bot, self.rotor_bot)
        self._remove_flag(claw_top, self.rotor_top)
        # Set new flags
        self._set_flag(claw_bot_next, self.rotor_bot)
        self._set_flag(claw_top_next, self.rotor_top)

    #
    # Returns true if a goal has been reached otherwise false
    #
    def goal_reached(self):
        if self._flag_is_set(self.tcp_pos, self.goal):
            return True
        else:
            return False

    #
    # Calculates and returns the reward for the last move made
    # Should only be executed before replacing the goal
    #
    def calculate_reward(self):
        # TODO: Check if space between claws is empty
        # Probably by drawing a line between the two claws and calculating all points between
        # Then return -20
        if self.goal_reached():
            if self.tcp_pos == self.last_goal:
                return 20
            else:
                return 10
        else:
            return -1

    #
    # Executes an action and returns the reward of that action
    #
    def make_move(self, action):
        # Execute action
        try:
            if action == Action.UP:
                self.move_up()
            elif action == Action.DOWN:
                self.move_down()
            elif action == Action.LEFT:
                self.move_left()
            elif action == Action.RIGHT:
                self.move_right()
            elif action == Action.ROTATE_CLOCKWISE:
                self.turn_clockwise()
            elif action == Action.ROTATE_COUNTER_CLOCKWISE:
                self.turn_counter_clockwise()
        # Catch exceptions
        except ClawOutOfWorldException:
            return -10
        except WireCollisionException:
            return -10

        # Calculate reward
        reward = self.calculate_reward()

        # Check if we need to set a new goal
        if self.goal_reached():
            # Find new goal
            new_goal = self.find_new_goal(action)
            # Remove old goal
            self._remove_flag(self.tcp_pos, self.goal)
            # Set new goal
            self._add_flag(new_goal, self.goal)

        return reward

    #
    # Runs a few tests
    #
    def test(self):
        print(self.__world.shape)
        # Check if in world states work
        in_world_coords = [math.ceil(self.__world.shape[0] / 2 - self.__state_size / 2), 0]
        print("Check if in world coords work: ", in_world_coords)
        print(self.get_state(in_world_coords))

        # Check if out of world x works
        out_of_world_x_coords = [self.__world.shape[0] - self.__state_size,
                                 self.__world.shape[1] - self.__state_size + 1]
        print("Check if out of world x coords work: ", out_of_world_x_coords)
        print(self.get_state(out_of_world_x_coords))

        # Check if out of world y works
        out_of_world_y_coords = [self.__world.shape[0] - self.__state_size + 1,
                                 self.__world.shape[1] - self.__state_size]
        print("Check if out of world y coords work: ", out_of_world_y_coords)
        print(self.get_state(out_of_world_y_coords))

        # Check if out of world x and y works
        print("Check if out of world x and y works")
        out_of_world_y_coords = [self.__world.shape[0] - self.__state_size + 1, self.__world.shape[1] -
                                 self.__state_size + 1]
        print(self.get_state(out_of_world_y_coords))

        # Check if init state works
        init_state_coords = [0, math.floor(self.__world.shape[1] / 2 - self.__state_size / 2)]
        print("Check if init state works with coords: ", init_state_coords)
        print(self.get_state(init_state_coords))


class WireCollisionException(Exception):
    pass


class ClawFreedFromWireException(Exception):
    pass


class ClawOutOfWorldException(Exception):
    pass

if __name__ == '__main__':
    world = SimulatedWorld()
    world.test()
