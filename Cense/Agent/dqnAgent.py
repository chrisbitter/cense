import threading
import logging
from Cense.Trainer.gpu import GPU_Trainer
import matplotlib.pyplot as plt

from Cense.World.dummy_world import DummyWorld as World
# from Resources.PrioritizedExperienceReplay.rank_based import Experience
from Cense.World.dummy_world import TerminalStateError

import Cense.NeuralNetworkFactory.nnFactory as factory
import time
import numpy as np
import os
from keras.models import model_from_json

#silence tf compile warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class DeepQNetworkAgent(object):
    # simulated_world = None
    real_world = None
    model_file = "../../Resources/nn-data/model.json"
    weights_file = "../../Resources/nn-data/weights.h5"
    train_switch_file = "../../Resources/train_switch"
    # image_path = ""
    original_epsilon = 0
    epsilon = 0
    gamma = 0
    # lookup_file = ""

    experienceBuffer = None
    lock_buffer = threading.Lock()

    working_collectors = 0

    def __init__(self, use_old_model=False):

        project_root_folder = os.path.join(os.getcwd(), "..", "..", "")

        # use the real world
        self.world = World()

        # if there's already a model, use it. Else create new model
        if use_old_model and os.path.isfile(self.model_file) and os.path.isfile(self.weights_file):
            with open(self.model_file) as file:
                model_config = file.readline()
                self.model = model_from_json(model_config)

            self.model.load_weights(self.weights_file)

        else:
            self.model = factory.model_dueling(self.world.STATE_DIMENSIONS, self.world.ACTIONS)

            #todo: check if keras lambda save is fixed
            #with open(self.model_file, 'w') as file:
            #    file.write(self.model.to_json())

            self.model.save_weights(self.weights_file)

        self.trainer = GPU_Trainer(project_root_folder)

        tic = time.time()
        self.trainer.send_model_to_gpu()
        toc = time.time()
        print("Send model to GPU: ", toc-tic)

    def train(self, exploration_probability=.1, runs_before_update=10):
        print("train")

        open(self.train_switch_file, 'w')

        states = []
        actions = []
        rewards = []
        suc_states = []
        terminals = []

        # statistics
        runs = []

        while os.path.isfile(self.train_switch_file):

            # start new run
            print("New run...")

            # statistics
            run_reward = 0
            run_steps = 0

            self.world.init_nonterminal_state()

            state, terminal = self.world.observe()

            while not terminal:
                # evaluate policy and value

                q_values = self.model.predict(np.expand_dims(state, axis=0))

                # explore with exploration_probability, else exploit
                if np.random.random() < exploration_probability:
                    action = np.random.randint(self.world.ACTIONS)
                else:
                    action = np.argmax(q_values)

                try:
                    suc_state, reward, terminal = self.world.execute(action)
                except TerminalStateError as e:
                    print(e)
                    break

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                suc_states.append(suc_state)
                terminals.append(terminal)

                # collect stats
                run_steps += 1
                run_reward += reward

            if run_steps:
                print("Run: ", run_steps, "steps")
                runs.append([run_steps, run_reward])

            # train nn after collecting some experience
            if len(runs) % runs_before_update == 0:
                logging.debug("update net")

                tic = time.time()
                self.trainer.send_experience_to_gpu(states, actions, rewards, suc_states, terminals)
                toc = time.time()
                print("Send experience: ", toc - tic)

                self.trainer.train_on_gpu()

                tic = time.time()
                self.trainer.fetch_model_weights_from_gpu()
                toc = time.time()
                print("Fetch weights: ", toc - tic)

                self.model.load_weights(self.weights_file)

                states = []
                actions = []
                rewards = []
                suc_states = []
                terminals = []

        plt.figure(1)

        # steps per run
        plt.subplot(221)
        plt.plot(range(1, len(runs) + 1), [run[0] for run in runs])
        plt.xlabel('run')
        plt.title('steps')

        # reward per run
        plt.subplot(222)
        plt.plot(range(1, len(runs) + 1), [run[1] for run in runs])
        plt.xlabel('run')
        plt.title('rewards')
        plt.show()

        print("Stop training")


if __name__ == '__main__':
    print("Starting from dqnAgent")
    agent = DeepQNetworkAgent()

    agent.train(10)
