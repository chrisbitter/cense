import numpy as np
import sys
from World.Simulated.simulatedWorld import SimulatedWorld
from Decider.Lookup.lookupDecider import LookupDecider


class CenseAgent:
    world = None
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
        self.world = SimulatedWorld(image_path)
        self.decider = LookupDecider(epsilon, gamma, lookup_file)

    def train(self, epochs, init, verbose=False):
        # If this game isn't played from scratch, find a stable state
        if not init:
            self.decider.set_epsilon(0)
            stable_state, stable_tcp = self.play(verbose=True)
            if type(stable_state) is not np.ndarray:
                init = True
            self.decider.set_epsilon(self.epsilon)
        score = 0
        moves = 0
        final_goal_reached = False
        for i in range(epochs):
            self.world = SimulatedWorld(self.image_path)
            # Introduce the final state into the world
            if not init:
                state = stable_state
                tcp = np.copy(stable_tcp).tolist()
                self.world.move_tcp_and_change_world(tcp, state)
            status = 1
            print("Starting epoch\n ____________________________________________________ \n\n\n")
            while status == 1:
                # Get old state
                state_old = self.world.get_state_by_tcp()
                # Decide the move
                action = self.decider.decide(state_old)
                if verbose:
                    print("Decided to take action: " + str(action.name))
                # Execute the move and get the reward
                reward = self.world.make_move(action)
                score += reward
                moves += 1
                if verbose:
                    print("Executed action, reward is: " + str(reward))
                # Get new state
                state_new = self.world.get_state_by_tcp()
                if verbose:
                    print("World state is now:\n" + str(state_new))
                    print("Now at tcp: " + str(self.world.tcp_pos))
                # Update q values
                self.decider.update_q(state_old, state_new, action, reward)
                # Check if reward indicates a final state
                if reward in [20, -10, -20]:
                    if reward == 20:
                        final_goal_reached = True
                    status = 0
                print("\n")
            # Update epsilon values at the end of each round
            if self.epsilon > 0.1:
                self.epsilon -= (1 / epochs)
                print("Updating epsilon value to: " + str(self.epsilon))
                self.decider.set_epsilon(self.epsilon)
        print("Final score: " + str(score) + " in " + str(moves) + " moves \nscore per move is: " + str(score/moves) +
              " \nFinal goal reached: " + str(final_goal_reached) + "\n" + str(moves/epochs) + " moves per epoch")

    def play(self, verbose):
        rewards = []
        state = self.world.get_state_by_tcp()
        status = 1
        stable_state = None
        stable_tcp = None
        while True:
            # Decide on the action
            action = self.decider.decide(state)
            if verbose:
                print("Decided to take action: " + str(action.name))
            # Taking action
            reward = self.world.make_move(action)
            if verbose:
                print("Executed action, reward is: " + str(reward))
            rewards.append(reward)
            state = self.world.get_state_by_tcp()
            if verbose:
                print("Now at tcp: " + str(self.world.tcp_pos))

            # Check if a goal was reached if so take new state as stable state
            if reward == 10:
                print("Stable state found: ")
                stable_state = np.copy(self.world.get_state_array_by_tcp())
                print(str(stable_state))
                stable_tcp = np.copy(self.world.tcp_pos)
                print("at: " + str(stable_tcp) + ", " + str(self.world.get_current_state_coordinates()))
            # If reward indicates a final state, stop the game
            if reward in [20, -10, -20]:
                status = 0
            # Stop game after more than 5 moves without goal
            if len(rewards) > 5:
                if 10 not in rewards[::-1][:5]:
                    print("Game lost; too many moves.")
                    break
            if not status == 1:
                break
        if stable_tcp is None:
            return 0, 0
        else:
            print("returning stable state: " + str(stable_tcp) + "\n\n")
            return stable_state, stable_tcp.tolist()


# An example of how the Agent class should be used
if __name__ == '__main__':
    agent = CenseAgent(sys.path[2] + '/Resources/wires/Cense_wire_01.png', 0, 0.5, "C:/Users/ls440063/lookup.json")
    agent.train(100, init=False, verbose=True)
    # print(str(agent.play(verbose=True)))
    agent.decider.persist_lookup_table()
