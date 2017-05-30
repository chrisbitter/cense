import threading
import logging
from Cense.Trainer.gpu import GPU_Trainer
import matplotlib.pyplot as plt
import json

from Cense.World.Real.realWorld import RealWorld as World
# from Resources.PrioritizedExperienceReplay.rank_based import Experience
from Cense.World.Real.realWorld import TerminalStateError

import Cense.NeuralNetworkFactory.nnFactory as Factory
import time
import numpy as np
import os
from keras.models import model_from_json

# silence tf compile warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class DeepQNetworkAgent(object):
    # simulated_world = None
    real_world = None
    model_file = "../../Resources/nn-data/model.json"
    weights_file = "../../Resources/nn-data/weights.h5"
    train_parameters = "../../Resources/train_parameters.json"
    # image_path = ""
    original_epsilon = 0
    epsilon = 0
    gamma = 0
    # lookup_file = ""

    experienceBuffer = None
    lock_buffer = threading.Lock()

    working_collectors = 0

    def __init__(self):

        project_root_folder = os.path.join(os.getcwd(), "..", "..", "")

        # use the real world
        self.world = World()

        with open(self.train_parameters) as json_data:
            config = json.load(json_data)

        # if there's already a model, use it. Else create new model
        if config["use_old_model"] and os.path.isfile(self.model_file) and os.path.isfile(self.weights_file):
            with open(self.model_file) as file:
                model_config = file.readline()
                self.model = model_from_json(model_config)

            self.model.load_weights(self.weights_file)

        else:
            self.model = Factory.model_dueling(self.world.STATE_DIMENSIONS, self.world.ACTIONS)

            # todo: check if keras lambda save is fixed
            # with open(self.model_file, 'w') as file:
            #    file.write(self.model.to_json())

            self.model.save_weights(self.weights_file)

        self.trainer = GPU_Trainer(project_root_folder)

        self.trainer.send_model_to_gpu()

    def train(self):

        with open(self.train_parameters) as json_data:
            config = json.load(json_data)

        exploration_probability = config["exploration_probability"]
        runs_before_update = config["runs_before_update"]

        print("Train with exploration probability ", exploration_probability, "and updates after", runs_before_update,
              "runs")

        states = []
        actions = []
        rewards = []
        suc_states = []
        terminals = []

        # statistics
        runs = []
        run_number = 1

        fig = plt.figure()
        # steps per run
        plt.subplot(221)
        plt.xlabel('run')
        plt.title('steps')
        steps_plot, = plt.plot([], [])
        steps_ax = plt.gca()

        plt.subplot(222)
        plt.xlabel('run')
        plt.title('reward')
        rewards_plot, = plt.plot([], [])
        rewards_ax = plt.gca()

        plt.subplot(223)
        plt.xlabel('action')
        plt.ylabel('q-value')
        #plt.bar
        bar_plot = plt.bar(list(range(self.world.ACTIONS)), np.zeros(self.world.ACTIONS))
        bar_ax = plt.gca()

        self.world.reset()

        while True:

            # start new run
            with open(self.train_parameters) as json_data:
                config = json.load(json_data)

            if exploration_probability != config["exploration_probability"]:
                exploration_probability = config["exploration_probability"]
                print("Using new exploration probability: ", exploration_probability)

            if runs_before_update != config["runs_before_update"]:
                runs_before_update = config["runs_before_update"]
                print("Now updating net every", runs_before_update, "run")

            if not config["do_train"]:
                break

            # statistics
            run_reward = 0
            run_steps = 0

            state, terminal = self.world.observe_state(), self.world.in_terminal_state()

            while not terminal:
                # evaluate policy and value

                # plt.imshow(state)
                # plt.pause(0.001)



                # explore with exploration_probability, else exploit
                if np.random.random() < exploration_probability:
                    action = np.random.randint(self.world.ACTIONS)
                else:
                    q_values = self.model.predict(np.expand_dims(state, axis=0))

                    for rect, q_val in zip(bar_plot, q_values[0]):
                        rect.set_height(q_val)

                    bar_ax.relim()
                    bar_ax.autoscale_view()
                    plt.draw()
                    plt.pause(.001)

                    action = np.argmax(q_values)

                try:
                    suc_state, reward, terminal = self.world.execute(action)

                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)
                    suc_states.append(suc_state)
                    terminals.append(terminal)

                    # collect stats
                    run_steps += 1
                    run_reward += reward

                except TerminalStateError as e:
                    #apperantly the last action already resulted in touching the wire which wasn't caught
                    print("Already in Terminal State.")
                    if len(states):
                        rewards[-1] = e.args[1]
                        terminals[-1] = True

                        # replace last reward with reward proposed by exception
                        run_reward -= reward
                        run_reward += e.args[1]
                    break

            self.world.reset()

            if run_steps:
                print("Run: ", run_number, "\n\t", "steps:", run_steps, "\n\t", "reward:", run_reward)

                steps_plot.set_xdata(np.append(steps_plot.get_xdata(), [run_number]))
                steps_plot.set_ydata(np.append(steps_plot.get_ydata(), [run_steps]))
                rewards_plot.set_xdata(np.append(rewards_plot.get_xdata(), [run_number]))
                rewards_plot.set_ydata(np.append(rewards_plot.get_ydata(), [run_reward]))

                steps_ax.relim()
                steps_ax.autoscale_view()
                rewards_ax.relim()
                rewards_ax.autoscale_view()

                plt.draw()
                plt.pause(.001)
                run_number += 1

            # train nn after collecting some experience
            if run_steps and run_number % runs_before_update == 0:
                print("Train on GPU")
                # logging.debug("update net")

                tic = time.time()
                self.trainer.send_experience_to_gpu(states, actions, rewards, suc_states, terminals)
                print("\tSend experience: ", time.time() - tic)

                tic = time.time()
                self.trainer.train_on_gpu()
                print("\tTrain on GPU:", time.time() - tic)

                tic = time.time()
                self.trainer.fetch_model_weights_from_gpu()
                print("\tGet weights: ", time.time() - tic)

                self.model.load_weights(self.weights_file)

                states = []
                actions = []
                rewards = []
                suc_states = []
                terminals = []

        print("Stop training")

        plt.show()


if __name__ == '__main__':
    print("Starting from dqnAgent")
    agent = DeepQNetworkAgent()

    agent.train()
