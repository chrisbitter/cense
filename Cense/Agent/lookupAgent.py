import sys

import numpy as np

# from World.Real.realWorld import RealWorld
from Resources.misc.Decider.Lookup.lookupDecider import LookupDecider
from Resources.misc.Simulated.simulatedWorld import SimulatedWorld


class AlreadyTrainedException(Exception):
    pass


class LookupAgent:
    simulated_world = None
    real_world = None
    decider = None
    image_path = ""
    original_epsilon = 0
    epsilon = 0
    gamma = 0
    lookup_file = ""

    def __init__(self, image_path, epsilon, gamma, lookup_file):
        self.image_path = image_path
        self.epsilon = epsilon
        self.original_epsilon = epsilon
        self.gamma = gamma
        self.lookup_file = lookup_file
        self.simulated_world = SimulatedWorld(image_path)
        self.decider = LookupDecider(epsilon, gamma, lookup_file)
        # self.real_world = RealWorld()

    #
    # Trains the decider in a number of epochs
    # init tells the method whether it should try to fetch a stable state or not
    #
    def train(self, epochs_l, init, verbose=False):
        # Initializing the state variables so pycharm will stop bugging me about it
        stable_state = None
        stable_tcp = None
        stable_action = None

        # If this game isn't played from scratch, find a stable state
        if not init:
            # Disable random actions
            self.decider.set_epsilon(0)
            # Get stable state
            stable_state, stable_tcp, stable_action = self.play(verbose=False)

            """
            try:
                stable_state, stable_tcp, stable_action = self.play(verbose=False)
            except AlreadyTrainedException:
                # Already able to play this wire, try to train it from the beginning
                init = True
                if verbose:
                    print("Got stable state")
            """

            # Couldn't find a stable state, force initialization
            if type(stable_state) is not np.ndarray:
                init = True
            # Reset random action probability
            self.decider.set_epsilon(self.epsilon)

        # Initialize stats
        score = 0
        moves = 0
        final_goal_reached = False
        furthest_x = 0

        print("training wire: " + self.image_path)

        # Run over all epochs
        for iterator in range(epochs_l):
            self.simulated_world = SimulatedWorld(self.image_path)

            # Introduce the final state into the world
            if not init:
                state = stable_state
                tcp = np.copy(stable_tcp).tolist()
                self.simulated_world.move_tcp_and_change_world(tcp, state, stable_action)

            if verbose:
                print("Starting epoch\n ____________________________________________________ \n\n\n")
            status = 1
            while status == 1:
                # Get old state
                state_old = self.simulated_world.get_state_by_tcp()

                # Decide the move
                action = self.decider.decide(state_old)
                # Execute the move and get the reward
                reward = self.simulated_world.make_move(action)

                # Get new state
                state_new = self.simulated_world.get_state_by_tcp()
                # Update q values
                self.decider.update_q(state_old, state_new, action, reward)

                # Give out some information
                if verbose:
                    print("Executed action, reward is:\t " + str(reward))
                    # print("World state is now:\n" + str(state_new))
                    print("Now at tcp: \t\t\t" + str(self.simulated_world.tcp_pos))

                # Update the statistics
                if self.simulated_world.tcp_pos[0] > furthest_x:
                    furthest_x = self.simulated_world.tcp_pos[0]
                score += reward
                moves += 1

                # Check if reward indicates a final state
                if reward in [20, -10, -20]:
                    if reward == 20:
                        final_goal_reached = True
                    status = 0
            if verbose:
                print("Epoch done \n\n\n")

            # Update epsilon values at the end of each round
            if self.epsilon > 0.1:
                self.epsilon -= (1 / epochs_l)
                self.decider.set_epsilon(self.epsilon)
                if verbose:
                    print("Updating epsilon value to: \t" + str(self.epsilon))
        # Print an overview of the stats
        if verbose:
            print("Final score: " + str(score) + " in " + str(moves) + " moves \nscore per move is: " + str(
                score / moves) + " \nFinal goal reached: " + str(final_goal_reached) + "\n" + str(moves / epochs_l)
                  + " moves per epoch " + "\nFurthest x: " + str(furthest_x))

    #
    # Plays the game based on the lookup table
    # Returns the Coordinates of the last goal it hit, and the associated state array
    # If no goal is hit, returns 0
    #
    def play(self, verbose):
        print("playing wire: " + self.image_path)
        rewards = []
        state = self.simulated_world.get_state_by_tcp()
        status = 1
        stable_state = None
        stable_tcp = None
        stable_action = None
        while True:
            # Decide on the action
            action = self.decider.decide(state)
            # Taking action
            reward = self.simulated_world.make_move(action)
            rewards.append(reward)
            state = self.simulated_world.get_state_by_tcp()
            if verbose:
                print("Executed action " + str(action.name) + " reward is: " + str(reward))
                print("Now at tcp: " + str(self.simulated_world.tcp_pos))

            # Check if a goal was reached if so take new state as stable state
            if reward in [10, 20]:
                stable_state = np.copy(self.simulated_world.get_state_array_by_tcp())
                stable_tcp = np.copy(self.simulated_world.tcp_pos)
                stable_action = self.simulated_world.last_directed_action
                if verbose:
                    print("Stable state found at: " + str(stable_tcp))

            # If reward indicates a final state, stop the game
            if reward in [20, -10, -20]:
                status = 0
                if reward == 20:
                    raise AlreadyTrainedException

            # Stop game after more than 5 moves without goal
            if len(rewards) > 5:
                if 10 not in rewards[::-1][:5]:
                    if verbose:
                        print("Game lost; too many moves.")
                    break

            # Check if game should be stopped
            if not status == 1:
                break
        # No stable tcp found, signal it with non numpy return
        if stable_tcp is None:
            return 0, 0, 0
        # Stable tcp found, return it
        else:
            if verbose:
                print("returning stable state: " + str(stable_tcp) + "\n\n")
            return stable_state, stable_tcp.tolist(), stable_action

    def play_on_machine(self, verbose):
        rewards = []
        state = self.simulated_world.get_state_by_tcp()
        status = 1

        while True:
            # Decide on the action
            action = self.decider.decide(state)
            # Taking action
            reward = self.simulated_world.make_move(action)
            rewards.append(reward)

            # If reward indicates a final state, stop the game
            if reward in [20, -10, -20]:
                status = 0
            # No collision detected, perform action with robot
            else:
                self.real_world.perform_action(action)
            state = self.simulated_world.get_state_by_tcp()
            if verbose:
                print("Executed action " + str(action.name) + " reward is: " + str(reward))
                print("Now at tcp: " + str(self.simulated_world.tcp_pos))

            # Stop game after more than 5 moves without goal
            if len(rewards) > 5:
                if 10 not in rewards[::-1][:5]:
                    print("Game lost; too many moves.")
                    break

            # Check if game should be stopped
            if not status == 1:
                break
        self.real_world.reset()

    #
    # Changes the Epsilon value in the decider and the agent
    #
    def set_epsilon(self, epsilon):
        self.epsilon = epsilon
        self.decider.set_epsilon(epsilon)

    #
    # Changes the current world, and changes the image_path
    #
    def set_wire_path(self, wire_path_l):
        self.image_path = wire_path_l
        self.simulated_world = SimulatedWorld(self.image_path)


# An example of how the Agent class should be used
if __name__ == '__main__':

    # Set values
    play = False
    iterations = 10
    epochs = 200
    random_action_probability = 0.1  # epsilon
    think_ahead_value = 0.5  # gamma
    print(sys.path)
    lookup_table_path = '../../Resources/lookup_tables/lookup.json'
    wire_base_path = '../../Resources/wires/Cense_wire_'
    wire_path = '../../Resources/wires/Cense_wire_01.png'

    # Initialize Agent
    agent = LookupAgent(wire_path, random_action_probability, think_ahead_value, lookup_table_path)

    # Iterate over all wires
    for j in range(1, 32):
        # Build new path
        wire_number = str(j)
        if j < 10:
            wire_number = "0" + str(j)
        wire_path = wire_base_path + wire_number + '.png'
        # Set path to the next wire
        agent.set_wire_path(wire_path)
        print("switching to wire: " + wire_path)

        # Play with set values on the current cable
        if play:
            iterations = 1
        for i in range(iterations):
            if play:
                agent.set_epsilon(0)
                try:
                    print(agent.play(verbose=True))
                except AlreadyTrainedException:
                    pass
                break
            # Start training on the current wire
            try:
                agent.train(epochs, init=False, verbose=True)
            except AlreadyTrainedException:
                break
            # Persist lookup table
            agent.decider.persist_lookup_table()
