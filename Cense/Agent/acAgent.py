import time

import yaml

import os.path as path

from Cense.NeuralNetworkFactory.nnFactory import model_ac as model

from Cense.World.Real.realWorld import RealWorld as World
from Cense.World.Real.realWorld import TerminalStateError
from Resources.PrioritizedExperienceReplay.rank_based import Experience
from Cense.World.Camera.camera import Camera

from keras.models import Model
import numpy as np
from Cense.Trainer.gpu import GPU_Trainer as Trainer


class AC_Agent(object):
    # simulated_world = None
    world = None
    gamma = 0

    model = None

    working_collectors = 0

    def __init__(self, learning_rate):

        self.learning_rate = learning_rate

        # use the real world
        self.world = World()

        #self.experienceBuffer = Experience()

        # todo init model and send to gpu
        self.model = model(self.world.get_state_dimensions(), self.world.get_action_dimensions())

        self.trainer = Trainer()

        self.train(10)

    def train(self, episodes, update_after_episode="False"):

        episode_states = []
        episode_actions = []
        episode_rewards = []
        episode_suc_states = []
        episode_terminals = []

        # statistics
        episode_reward = 0
        episode_steps = 0

        for episode in range(episodes):

            self.world.init_nonterminal_state()

            state, terminal = self.world.observe()

            while not terminal:
                episode_steps += 1

                # evaluate policy and value
                action_distribution, value = self.model.predict(state)

                # get action from distribution
                action = np.random.choice(range(6), action_distribution)

                try:
                    suc_state, reward, terminal = self.world.execute(action)
                except TerminalStateError as e:
                    print(e.message)
                    break

                episode_reward += reward

                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(reward)
                episode_suc_states.append(suc_state)
                episode_terminals.append(terminal)

            # update after last episode and also if network should be updated after every episode
            if update_after_episode | episode == episodes - 1:

                self.trainer.upload_to_gpu(episode_states, episode_actions, episode_rewards, episode_suc_states,
                                           episode_terminals)
                self.trainer.train_on_gpu()

                new_model_weights = self.trainer.fetch_model_weights_from_gpu()
                self.model.set_weights(new_model_weights)

                episode_states = []
                episode_actions = []
                episode_rewards = []
                episode_suc_states = []
                episode_terminals = []


if __name__ == '__main__':
    print("Starting from acAgent")
    agent = AC_Agent()
