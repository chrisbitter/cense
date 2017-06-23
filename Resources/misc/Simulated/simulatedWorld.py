import World.Simulated.worldParser as worldParser
import numpy as np
from World.world import World
import math
from Decider.action import Action
from World.state import State


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
    # Coordinates of the current goal
    current_goal = (0, 0)
    # Last directed action that was performed
    last_directed_action = None

    # Tested
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
    # Tested
    #
    def __create_init_state(self):
        # print("Initializing world")
        # Calculate final goal
        for i in range(self.__world.shape[1] - 1):
            if self._flag_is_set((self.__world.shape[0] - 1, i), self.wire):
                self.last_goal = (self.__world.shape[0] - 1, i)
        # Calculate the offset of the claws from the center point of a given state
        tool_offset = math.floor(self.__state_size / 2)

        # Calculate worlds center based on the assumption that the wire always starts in the middle of the image
        center = math.floor(self.__world.shape[1] / 2)
        for i in range(self.__world.shape[1]):
            if self._flag_is_set([0, i], self.wire):
                center = i
        self.tcp_pos = [0, center]
        if center < math.floor(self.__state_size / 2):
            print("Warning: image is to small to properly position claws")

        # Calculate initial claw positions based on the worlds center and the previously calculated offset
        upper_claw_pos = (0, (center + tool_offset))
        lower_claw_pos = (0, (center - tool_offset))
        self._add_flag(upper_claw_pos, self.rotor_top)
        self._add_flag(lower_claw_pos, self.rotor_bot)

        # Set initial goal
        self.current_goal = (tool_offset, center)
        self._add_flag(self.current_goal, self.goal)

    #
    # Sets a flag while maintaining all other flags that have been set
    # Tested
    #
    def _add_flag(self, position, state):
        if self.is_out_of_world(position):
            return
        if not self._flag_is_set(position, state):
            self.__world[position[0], position[1]] += state

    #
    # Removes a flag if it is set
    #
    def _remove_flag(self, position, state):
        if self.is_out_of_world(position):
            return
        if self._flag_is_set(position, state):
            self.__world[position[0], position[1]] -= state

    #
    # Sets a flag while removing all other set flags
    # Tested
    #
    def _set_flag(self, position, state):
        if self.is_out_of_world(position):
            return
        self.__world[position[0], position[1]] = state

    #
    # Function that checks whether a flag is set in a given position
    # Tested
    #
    def _flag_is_set(self, position, state):
        if self.is_out_of_world(position):
            return False
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

    def _flag_is_between_claws(self, flag):
        flag_found = False
        claws = self.find_claw_positions()
        # Check if we can put a linear function through the claws
        if not claws[0][0] == claws[1][0]:
            if claws[0][1] >= claws[1][1]:
                upper_claw = claws[0]
                lower_claw = claws[1]
            else:
                upper_claw = claws[1]
                lower_claw = claws[0]
            if claws[0][0] <= claws[1][0]:
                first_claw = claws[0]
                second_claw = claws[1]
            else:
                first_claw = claws[1]
                second_claw = claws[0]
            # print("upper claw: " + str(upper_claw))
            # print("lower claw: " + str(lower_claw))
            # calculate gradient between the claws
            m = (lower_claw[1] - upper_claw[1]) / (lower_claw[0] - upper_claw[0])
            # Set offset
            b = -(m*upper_claw[0]-upper_claw[1])
            # print("Function is: f(x)=" + str(m) + "*x + " + str(b))
            # Check if there is a point containing a wire
            for i in range(first_claw[0], second_claw[0]):
                if self._flag_is_set((i, round(m * i + b)), flag):
                    flag_found = True
                    break
        # If claws are placed above each other, search in a vertical line
        else:
            upper_claw = np.max([claws[0][1], claws[1][1]])
            lower_claw = np.min([claws[0][1], claws[1][1]])
            for i in range(lower_claw, upper_claw):
                if self._flag_is_set((claws[0][0], i), flag):
                    flag_found = True
                    break
        return flag_found

    #
    # Checks if a position is out world
    # Returns True if so, otherwise False
    #
    def is_out_of_world(self, position):
        if position[0] >= self.__world.shape[0] or position[0] < 0:
            return True
        if position[1] >= self.__world.shape[1] or position[1] < 0:
            return True
        return False

    #
    # returns coordinates of a new goal based on the coordinates of the old goal
    #
    def find_new_goal(self, previous_action):
        # Make a list of all wire positions around the old goal
        list_of_wires = []
        pos_list = []
        # Calculate upper left of the current state based on the tool center point position
        state_offset = self.get_current_state_coordinates()
        for i in range(self.__state_size):
            for j in range(self.__state_size):
                check_pos = [state_offset[0] + i, state_offset[1] + j]
                pos_list.append(check_pos)
                if self._flag_is_set(check_pos, self.wire):
                    list_of_wires.append([i, j])
        list_of_wires = np.array(list_of_wires)
        non_cleaned_list = np.copy(list_of_wires)

        # Clean list
        if previous_action == Action.UP:
            list_of_wires = list_of_wires[list_of_wires[:, 1] >= 2]
        if previous_action == Action.DOWN:
            list_of_wires = list_of_wires[list_of_wires[:, 1] <= 2]
        if previous_action == Action.LEFT:
            list_of_wires = list_of_wires[list_of_wires[:, 0] <= 2]
        if previous_action == Action.RIGHT:
            list_of_wires = list_of_wires[list_of_wires[:, 0] >= 2]

        # Pick the potential goal with the highest distance
        if len(list_of_wires) == 0:
            print("\n\n")
            print("wire list empty!")
            print("previous action: " + str(previous_action.name))
            print("Positions checked: \n" + str(pos_list))
            value_list = [[self.__world[position[0]][position[1]], 30] for position in pos_list]
            print("Corresponding values: \n" + str(value_list))
            print("Current tcp_pos: " + str(self.tcp_pos))
            print("State looks like: \n" + str(self.get_state_by_tcp()))
            list_of_wires = non_cleaned_list

        # Calculate distance from original point to each potential goal
        distance_list = []
        for wire in list_of_wires:
            x_dist = np.abs(self.tcp_pos[0] - (state_offset[0] + wire[0]))
            y_dist = np.abs(self.tcp_pos[1] - (state_offset[1] + wire[1]))
            dist = x_dist + y_dist
            distance_list.append(dist)

        goal_index = np.argmax(distance_list)
        goal_relative = list_of_wires[goal_index]
        goal = (goal_relative[0] + state_offset[0], goal_relative[1] + state_offset[1])
        return goal

    #
    # Returns a state Array with coordinates describing the lower left
    # Tested
    #
    def get_state_array(self, coordinates):
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
        out_of_world_left = False
        if state_start_x < 0:
            # Not enough space, extend with out of world
            out_of_world_left = True
            # Calculate the size of the state
            x_size = self.__state_size + state_start_x

        # Make sure the snapshot is within boundaries
        if x_size + state_start_x > world_size_x:
            # x_offset is too close to the border, would try to copy nonexistent indices, adjust size of copy
            x_size = world_size_x - state_start_x
        if y_size + state_start_y > world_size_y:
            # y_offset is too close to the border, would try to copy nonexistent indices, adjust size of copy
            y_size = world_size_y - state_start_y

        # if the state would be out of world on the left always start copying from the leftmost column
        if out_of_world_left:
            state_start_x = 0
        # Do the copy of the selected area
        state = self.__world[state_start_x:state_start_x + x_size, state_start_y:state_start_y + y_size]

        # Fill the resized area with out of world Flag
        if x_size < self.__state_size:
            # Create filler Array
            out_of_world_stack = np.zeros([self.__state_size - x_size, state.shape[1]], dtype=np.uint8)
            out_of_world_stack.fill(self.out_of_world)
            if not out_of_world_left:
                # X went out of world on the right, extend towards the right
                state = np.vstack((state, out_of_world_stack))
            else:
                # X went out of world on the left, extend towards the left
                state = np.vstack((out_of_world_stack, state))
        if y_size < self.__state_size:
            # Y went out of world, extend towards the bottom
            out_of_world_stack = np.zeros([state.shape[0], self.__state_size - y_size], dtype=np.uint8)
            out_of_world_stack.fill(self.out_of_world)
            state = np.hstack((state, out_of_world_stack))
        return np.copy(state)

    #
    # Returns the state with coordinates describing the upper left
    # Tested
    #
    def get_state(self, coordinates):
        return State(self.get_state_array(coordinates))

    #
    # Returns the state array around the tcp
    # Tested
    #
    def get_state_array_by_tcp(self):
        state_position = self.get_current_state_coordinates()
        return self.get_state_array(state_position)

    #
    # Returns the state around the tcp
    # Tested
    #
    def get_state_by_tcp(self):
        state_position = self.get_current_state_coordinates()
        return self.get_state(state_position)

    #
    # Calculates the coordinates of the lower left of the current state based on the tcp
    # Tested
    #
    def get_current_state_coordinates(self):
        x_coordinate = self.tcp_pos[0] - math.floor(self.__state_size / 2)
        y_coordinate = self.tcp_pos[1] - math.floor(self.__state_size / 2)
        coordinates = [x_coordinate, y_coordinate]
        return coordinates

    #
    # Moves the tcp one unit to it's right and adjusts the positions of the claws
    # Tested
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
    # Tested
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
    # Tested
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
    # Tested
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
    # Moves the tcp to a new location and copies the state into there
    #
    def move_tcp_and_change_world(self, tcp, state, previous_action):
        # Find current claw positions
        claw_pos = self.find_claw_positions()
        # Remove current goal and claw flags
        self._remove_flag(claw_pos[0], self.rotor_bot)
        self._remove_flag(claw_pos[1], self.rotor_top)
        self._remove_flag(self.current_goal, self.goal)
        # Move tcp
        self.tcp_pos = tcp
        # Set previous action
        self.last_directed_action = previous_action
        # Reconstruct state
        state_offset = self.get_current_state_coordinates()
        for i in range(self.__state_size):
            for j in range(self.__state_size):
                position = (state_offset[0] + i, state_offset[1] + j)
                # Copy state
                self._set_flag(position, state[i][j])
                # Simultaneously search for goal
                if self._flag_is_set(position, self.goal):
                    self.current_goal = position

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
        r_bot_idx = r_top_idx = 0
        for idx, pos in enumerate(self.positions_R_top):
            if np.array_equal(pos, claw_top):
                r_top_idx = idx
                break
        # Go to the next element and check for overflow
        if r_top_idx == self.positions_R_top.shape[0] - 1:
            claw_top_next = np.add(state_offset, self.positions_R_top[0])
        else:
            claw_top_next = np.add(state_offset, self.positions_R_top[r_top_idx + 1])

        # Find position in the position list for bottom claw
        for idx, pos in enumerate(self.positions_R_bot):
            if np.array_equal(pos, claw_bot):
                r_bot_idx = idx
                break
        # Go to the next element and check for overflow
        if r_bot_idx == self.positions_R_top.shape[0] - 1:
            claw_bot_next = np.add(state_offset, self.positions_R_bot[0])
        else:
            claw_bot_next = np.add(state_offset, self.positions_R_bot[r_bot_idx + 1])

        # Move the claws
        self.__move_claws(np.add(state_offset, claw_bot), claw_bot_next, np.add(state_offset, claw_top), claw_top_next)
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
        r_top_idx = r_bot_idx = 0
        for idx, pos in enumerate(self.positions_R_top):
            if np.array_equal(pos, claw_top):
                r_top_idx = idx
        # Go to the next element and check for overflow
        if r_top_idx == 0:
            claw_top_next = np.add(state_offset, self.positions_R_top[self.positions_R_top.shape[0] - 1])
        else:
            claw_top_next = np.add(state_offset, self.positions_R_top[r_top_idx - 1])

        # Find position in the position list for bottom claw
        for idx, pos in enumerate(self.positions_R_bot):
            if np.array_equal(pos, claw_bot):
                r_bot_idx = idx
        # Go to the next element and check for overflow
        if r_bot_idx == 0:
            claw_bot_next = np.add(state_offset, self.positions_R_bot[self.positions_R_top.shape[0] - 1])
        else:
            claw_bot_next = np.add(state_offset, self.positions_R_bot[r_bot_idx - 1])

        # Move the claws
        self.__move_claws(np.add(state_offset, claw_bot), claw_bot_next, np.add(state_offset, claw_top), claw_top_next)
        # No need to adjust tcp, because it stays the same

    #
    # Returns the position of the bottom claw and the top claw
    # Tested
    #
    def find_claw_positions(self):
        # Calculate upper left of the current state based on the tool center point position
        state_offset = self.get_current_state_coordinates()
        # Search for the lower and the upper claw
        rotor_top = rotor_bot = [0, 0]
        for i in range(self.__state_size):
            for j in range(self.__state_size):
                check_pos = [state_offset[0] + i, state_offset[1] + j]
                if self._flag_is_set(check_pos, self.rotor_bot):
                    rotor_bot = check_pos
                elif self._flag_is_set(check_pos, self.rotor_top):
                    rotor_top = check_pos
        return rotor_bot, rotor_top

    #
    # Moves the claws from the old positions to the new ones if it is safe to do so, otherwise raises an Exception
    # Tested
    #
    # noinspection PyUnresolvedReferences
    def __move_claws(self, claw_bot, claw_bot_next, claw_top, claw_top_next):
        # Check if new position would be out of map
        next_positions = [claw_top_next, claw_bot_next]

        # Upper boundaries
        for next_position in next_positions:
            if self.__world.shape[0] <= next_position[0]:
                raise ClawOutOfWorldException
            if self.__world.shape[1] <= next_position[1]:
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
        if self._flag_is_between_claws(self.goal):
            return True
        # Check all other spots between the tcps
        return False

    #
    # Calculates and returns the reward for the last move made
    # Should only be executed before replacing the goal
    #
    def calculate_reward(self):
        # Check if space between the claws is empty
        # No wire found between the claws, return -20, agent left the area it's supposed to operate in
        if not self._flag_is_between_claws(self.wire):
            return -20

        if self.goal_reached():
            if self.current_goal == self.last_goal:
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
                self.last_directed_action = action
            elif action == Action.DOWN:
                self.move_down()
                self.last_directed_action = action
            elif action == Action.LEFT:
                self.move_left()
                self.last_directed_action = action
            elif action == Action.RIGHT:
                self.move_right()
                self.last_directed_action = action
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
            if self.last_directed_action is None:
                self.last_directed_action = Action.RIGHT
            # Find new goal
            new_goal = self.find_new_goal(self.last_directed_action)
            # Remove old goal flag
            self._remove_flag(self.current_goal, self.goal)
            # Set new goal flag
            self._add_flag(new_goal, self.goal)
            # Set new current goal
            self.current_goal = new_goal

        return reward

    #
    # Runs a few tests
    #
    def test(self):
        print("World shape = " + str(self.__world.shape))
        # # Check if in world states work
        # in_world_coords = [math.ceil(self.__world.shape[0] / 2 - self.__state_size / 2), 0]
        # print("Check if in world coords work: ", in_world_coords)
        # print(self.get_state_array(in_world_coords))
        #
        # # Check if out of world x works
        # out_of_world_x_coords = [self.__world.shape[0] - self.__state_size,
        #                          self.__world.shape[1] - self.__state_size + 1]
        # print("Check if out of world x coords work: ", out_of_world_x_coords)
        # print(self.get_state_array(out_of_world_x_coords))
        #
        # # Check if out of world y works
        # out_of_world_y_coords = [self.__world.shape[0] - self.__state_size + 1,
        #                          self.__world.shape[1] - self.__state_size]
        # print("Check if out of world y coords work: ", out_of_world_y_coords)
        # print(self.get_state_array(out_of_world_y_coords))
        #
        # # Check if out of world x and y works
        # print("Check if out of world x and y works")
        # out_of_world_y_coords = [self.__world.shape[0] - self.__state_size + 1, self.__world.shape[1] -
        #                          self.__state_size + 1]
        # print(self.get_state_array(out_of_world_y_coords))
        #
        # # Check if init state works
        # init_state_coords = [0, math.floor(self.__world.shape[1] / 2 - self.__state_size / 2)]
        # print("Check if init state works with coords: ", init_state_coords)
        # print(self.get_state_array(init_state_coords))
        # print("\n")

        # Check if get state by tcp works
        print("\n")
        actions_to_perform = [Action.RIGHT, Action.RIGHT, Action.RIGHT, Action.RIGHT, Action.ROTATE_COUNTER_CLOCKWISE, Action.RIGHT,
                              Action.UP, Action.UP]
        for action in actions_to_perform:
            print("performing: " + action.name)
            print("reward is: " + str(self.make_move(action)))
            print("\ntcp is: " + str(self.tcp_pos) + "\n")
            print(str(self.get_state_by_tcp()))
            print("\n")
        self.move_right()
        self.turn_clockwise()
        # print(str(self.get_state_by_tcp()) + "\n")
        # print("reward : " + str(self.calculate_reward()))
        # print(self.get_state_by_tcp())
        # print(self.find_new_goal(Action.RIGHT))


class WireCollisionException(Exception):
    pass


class ClawFreedFromWireException(Exception):
    pass


class ClawOutOfWorldException(Exception):
    pass


if __name__ == '__main__':
    world = SimulatedWorld()
    world.test()
