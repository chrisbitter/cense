from World.Simulated.simulatedWorld import SimulatedWorld
from Decider.Lookup.lookupDecider import LookupDecider
import numpy as np


class Agent:
    world = None
    decider = None
    image_path = ""
    original_epsilon = 0
    epsilon = 0
    gamma = 0
    lookup_file = ""

    if __name__ == '__main__':
        print("test")

    if __name__ == '_init':
        print("init")

    def init(self, image_path, epsilon, gamma, lookup_file):
        self.image_path = image_path
        self.epsilon = epsilon
        self.original_epsilon = epsilon
        self.gamma = gamma
        self.lookup_file = lookup_file
        self.world = SimulatedWorld(image_path)
        self.decider = LookupDecider(epsilon, gamma, lookup_file)

    def train(self, epochs, decider, init):
        if not init:
            stable_state, stable_tcp = self.play(decider, verbose=False)

        for i in range(epochs):
            world = SimulatedWorld(self.image_path)
            if not init:
                state = stable_state
                tcp = stable_tcp
            status = 1
            while status == 1:
                world.move_tcp_and_change_world(tcp, state)
                state_old = world.get_state_by_tcp()

                action = decider.decide(world.get_state_by_tcp())
                reward = world.make_move(action)
                state_new = world.get_state_by_tcp()
                new_center_coord = world.tcp_pos

                decider.update_q(state_old, state_new, action, reward)
                # if i % 100 == 0:
                #     print
                #     'Game {0}, action {1}, reward {2}'.format(i, code_actions[action], reward)
                state = np.copy(state_new)
                tcp = new_center_coord
                if reward in [20, -10, -20]:
                    status = 0
            if self.epsilon > 0.1:
                self.epsilon -= (1 / epochs)
                decider.set_epsilon(self.epsilon)

    def play(self, decider, verbose):
        rewards = []
        i = 0

        state = self.world.get_state_by_tcp()
        # disp_state(state, center_coord, action=None, reward=None, i=i, init=True)
        status = 1
        while status == 1:
            action = decider.decide(state)
            if verbose:
                print(action)
                print('Move %s; Taking action: %s' % (i + 1, action))
            reward = self.world.make_move(action)
            new_state = self.world.get_state_by_tcp()
            rewards.append(reward)
            if reward == 10:
                stable_state = np.copy(new_state)
                stable_tcp = self.world.tcp_pos
            # if verbose:
            #     disp_state(new_state, new_center_coord, action, reward, i + 1)
            # if verbose:
            #     print("Reward: %s" % (reward,))
            if reward in [20, -10, -20]:
                status = 0
            state = np.copy(new_state)
            i += 1
            if len(rewards) > 5:
                if 10 not in rewards[::-1][:5]:
                    print("Game lost; too many moves.")
                    break
        return stable_state, stable_tcp
