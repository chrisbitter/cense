import threading
import logging
from Cense.Trainer.gpu import GPU_Trainer
import matplotlib.pyplot as plt
import json

import math
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

        self.model = Factory.model_dueling(self.world.STATE_DIMENSIONS, self.world.ACTIONS)

        # if there's already a model, use it. Else create new model
        if config["use_old_model"] and os.path.isfile(self.weights_file):
            #with open(self.model_file) as file:
            #    model_config = file.readline()
            #    self.model = model_from_json(model_config)
            self.model.load_weights(self.weights_file)

        else:
            # todo: check if keras lambda save is fixed
            # with open(self.model_file, 'w') as file:
            #    file.write(self.model.to_json())
            self.model.save_weights(self.weights_file)

        self.trainer = GPU_Trainer(project_root_folder)

        self.trainer.send_model_to_gpu()

    def train(self):

        with open(self.train_parameters) as json_data:
            config = json.load(json_data)

        exploration_probability = config["exploration_probability_start"]
        runs_before_update = config["runs_before_update"]

        print("Train with exploration probability ", exploration_probability, "and updates after", runs_before_update,
              "runs")

        states = []
        actions = []
        rewards = []
        suc_states = []
        terminals = []

        # statistics
        # runs = []
        run_number = 1

        # steps per run
        plt.subplot(231)
        plt.xlabel('run')
        plt.title('steps')
        steps_plot, = plt.plot([], [])
        steps_ax = plt.gca()

        plt.subplot(232)
        plt.xlabel('run')
        plt.title('reward')
        rewards_plot, = plt.plot([], [])
        rewards_ax = plt.gca()

        plt.subplot(233)
        plt.xlabel('action')
        plt.ylabel('q-value')
        # plt.bar
        bar_plot = plt.bar(list(range(self.world.ACTIONS)), np.zeros(self.world.ACTIONS))
        bar_ax = plt.gca()

        plt.subplot(234)
        plt.xlabel('run')
        plt.title('exploration probability')
        exploration_plot, = plt.plot([], [])
        exploration_ax = plt.gca()

        plt.subplot(235)
        cam_view = plt.imshow(np.zeros(self.world.STATE_DIMENSIONS), cmap='gray')
        cam_view.norm.vmax = 1

        while True:

            # start new run
            with open(self.train_parameters) as json_data:
                config = json.load(json_data)

            if not config["do_train"]:
                break

            # statistics
            run_reward = 0
            run_steps = 0

            self.world.init_nonterminal_state()

            state, terminal = self.world.observe_state(), self.world.in_terminal_state()

            while not terminal:

                q_values = self.model.predict(np.expand_dims(state, axis=0))

                if np.any(np.isnan(q_values)):
                    raise ValueError("Net is broken!")

                for rect, q_val in zip(bar_plot, q_values[0]):
                    rect.set_height(q_val)
                    rect.set_color('b')

                # explore with exploration_probability, else exploit
                if np.random.random() < exploration_probability:
                    action = np.random.randint(self.world.ACTIONS)
                else:
                    action = np.argmax(q_values)

                if action == np.argmax(q_values):
                    bar_plot[action].set_color('g')
                else:
                    bar_plot[action].set_color('r')
                bar_ax.relim()
                bar_ax.autoscale_view()

                cam_view.set_data(state)
                plt.draw()
                plt.pause(.001)

                try:
                    suc_state, reward, terminal = self.world.execute(action)

                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)
                    suc_states.append(suc_state)
                    terminals.append(terminal)

                    state = suc_state

                    # collect stats
                    run_steps += 1
                    run_reward += reward

                except TerminalStateError as e:
                    # apperantly the last action already resulted in touching the wire which wasn't caught
                    print("Already in Terminal State.")
                    if len(states):
                        # correct collected experience, since last action led to terminal state
                        rewards[-1] = e.args[1]
                        terminals[-1] = True

                        # replace last reward with reward proposed by exception
                        run_reward -= reward
                        run_reward += e.args[1]
                    break

            if run_steps:
                print("Run: ", run_number, "\n\t", "steps:", run_steps, "\n\t", "reward:", run_reward)

                # plot statistics
                steps_plot.set_xdata(np.append(steps_plot.get_xdata(), [run_number]))
                steps_plot.set_ydata(np.append(steps_plot.get_ydata(), [run_steps]))
                rewards_plot.set_xdata(np.append(rewards_plot.get_xdata(), [run_number]))
                rewards_plot.set_ydata(np.append(rewards_plot.get_ydata(), [run_reward]))
                exploration_plot.set_xdata(np.append(exploration_plot.get_xdata(), [run_number]))
                exploration_plot.set_ydata(np.append(exploration_plot.get_ydata(), [exploration_probability]))

                steps_ax.relim()
                steps_ax.autoscale_view()
                rewards_ax.relim()
                rewards_ax.autoscale_view()
                exploration_ax.relim()
                exploration_ax.autoscale_view()

                plt.draw()
                plt.pause(.001)

                # train neural network after collecting some experience
                if run_number % runs_before_update == 0:
                    print("Train on GPU")
                    # logging.debug("update net")

                    self.trainer.send_experience_to_gpu(states, actions, rewards, suc_states, terminals)
                    self.trainer.train_on_gpu()
                    self.trainer.fetch_model_weights_from_gpu()

                    self.model.load_weights(self.weights_file)

                    # clear experience collection
                    states = []
                    actions = []
                    rewards = []
                    suc_states = []
                    terminals = []

                # decrease exploration every C steps
                if run_number % config["exploration_probability_update_steps"] == 0:
                    exploration_probability = max(
                        exploration_probability * config["exploration_probability_update_factor"],
                        config["exploration_probability_end"])

                if run_number % config["runs_before_test"] == 0:
                    print("Show progress!")
                    self.world.reset()

                    state, terminal = self.world.observe_state(), self.world.in_terminal_state()
                    run_steps = 0
                    run_reward = 0

                    # run until terminal state is reached/wire is touched
                    while not terminal:

                        q_values = self.model.predict(np.expand_dims(state, axis=0))

                        if np.any(np.isnan(q_values)):
                            raise ValueError("Net is broken!")

                        for rect, q_val in zip(bar_plot, q_values[0]):
                            rect.set_height(q_val)
                            rect.set_color('y')

                        action = np.argmax(q_values)

                        bar_plot[action].set_color('g')
                        bar_ax.relim()
                        bar_ax.autoscale_view()

                        cam_view.set_data(state)
                        plt.draw()
                        plt.pause(.001)

                        try:

                            suc_state, reward, terminal = self.world.execute(action)

                            states.append(state)
                            actions.append(action)
                            rewards.append(reward)
                            suc_states.append(suc_state)
                            terminals.append(terminal)

                            state = suc_state

                            # collect stats
                            run_steps += 1
                            run_reward += reward

                        except TerminalStateError as e:
                            # apperantly the last action already resulted in touching the wire which wasn't caught
                            print("Already in Terminal State.")

                            # replace last reward with reward proposed by exception
                            run_reward -= reward
                            run_reward += e.args[1]
                            break

                    if self.world.is_at_goal():
                        self.world.reset_current_start_pose()
                    else:
                        # ideally, this only reverses the last move and restores the last nonterminal state
                        # if not, we're back to the old start pose
                        self.world.init_nonterminal_state()
                        self.world.update_current_start_pose()

                    print("\tAchieved reward", run_reward, "in", run_steps, "steps!")
                    print("Continuing with learning")

                run_number += 1

        print("Stop training")

        statistics = np.dstack((steps_plot.get_ydata(), rewards_plot.get_ydata()))

        statistics = statistics.reshape(statistics.shape[1:])

        np.savetxt(time.strftime("%Y%m%d-%H%M%S") + ".csv", statistics, header="steps,reward")

        plt.show()


if __name__ == '__main__':
    print("Starting from dqnAgent")
    agent = DeepQNetworkAgent()

    agent.train()
