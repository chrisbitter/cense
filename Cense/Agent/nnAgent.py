import threading
import time

import yaml

import os.path as path

from Cense.Decider.NeuralNetwork.dqnDecider import DqnDecider
from Cense.Decider.NeuralNetwork.acDecider import AcDecider as Decider

from Cense.World.Real.realWorld import RealWorld as World
from Resources.PrioritizedExperienceReplay.rank_based import Experience
from Cense.World.Camera.camera_pygame import Camera

from keras.models import Model


class NeuralNetworkAgent(object):
    # simulated_world = None
    real_world = None
    decider = None
    # image_path = ""
    original_epsilon = 0
    epsilon = 0
    gamma = 0
    # lookup_file = ""

    experienceBuffer = None
    lock_buffer = threading.Lock()

    working_collectors = 0

    def __init__(self, learning_rate):

        self.learning_rate = learning_rate

        # use the real world
        self.world = World()
        self.decider = Decider()

        self.experienceBuffer = Experience()

        self.train(10)



    def train(self, epochs, infinite_training=False):

        try:
            while True:

                trainer = threading.Thread(target=self.trainer)
                collector = threading.Thread(target=self.collector(epochs))

                collector.start()
                trainer.start()

                collector.join()
                trainer.join()

                if not infinite_training:
                    break
        except KeyboardInterrupt:
            print("Abort Training")
        finally:
            # TODO: save everything
            pass

    def trainer(self, epochs=1, minibatch_size=5, copy_network_ratio=.8):

        timeout = int(5)
        current_timeout = 0

        # get current model
        with self.lock_model:
            model_config = self.current_model_config

        model = Model.from_config(model_config)
        model_target = Model.from_config(model_config)

        while True:

            with self.lock_buffer:
                samples, importance_sampling_weight, element_id = self.experienceBuffer.sample(51)

            if samples is not None & importance_sampling_weight is not None & element_id is not None:
                # sampling was successful is good

                # todo: learn
                # todo: update model_parameters

                # if terminal:
                #   target = reward
                # else:
                #   targets = rewards + discount_factor * model_target.value(successor_state) - model_target(state)

                # calculate targets: y = r + discount * Q_copy(
                y = sample[2] + model_copy

                pass
            else:
                # buffer didn't sample. Wait a certain timeout and try sampling again
                current_timeout = timeout

            # if there is no more collectors making experiences, stop refining the model
            if self.working_collectors <= 0:
                break

            # sleep for 5 seconds before checking Buffer again
            time.sleep(current_timeout)
            # reset timeout
            current_timeout = 0

    def work(self, max_episode_length,gamma):

    def collect_experience(self, episodes, exploration_probability, decider):

            self.world.init_nonterminal_state()

            # observe initial state (discard reward)
            state, _, terminal = self.world.observe_state(None)

            while not terminal:
                action = decider.decide(state)

                successor_state, reward, terminal = self.world.execute(action)

                experience = (state, action, reward, successor_state)

                # update buffer with new experience
                with self.lock_buffer:
                    self.experienceBuffer.store(experience)

                state = successor_state
        return

    def play(self, decider, verbose):
        with self.lock_model_config:
            if self.current_model_config is None:
                raise ValueError("No model configuration available")
            else:
                model = Model.from_config(self.current_model_config)

        try:
            self.world.init_nonterminal_state()

            state = self.world.observe_state()

            while not self.world.in_terminal_state():
                action = model(state)

                state, reward = self.world.execute(action)

        except KeyboardInterrupt:
            print("Abort Training")
        finally:
            # TODO: save everything
            pass


if __name__ == '__main__':
    print("Starting from nnAgent")
    agent = NeuralNetworkAgent()
