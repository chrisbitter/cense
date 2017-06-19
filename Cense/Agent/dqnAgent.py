import threading
import logging
from Cense.Trainer.gpu import GPU_Trainer
import matplotlib.pyplot as plt
import json
import csv

import math
from Cense.World.Real.realWorld import RealWorld as World
# from Resources.PrioritizedExperienceReplay.rank_based import Experience
from Cense.World.Real.realWorld import TerminalStateError, InsufficientProgressError

import Cense.NeuralNetworkFactory.nnFactory as Factory
from Cense.Visualization.visualization import TrainingVisualization
import time
import numpy as np
import os, stat
from keras.models import model_from_json

# silence tf compile warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class DeepQNetworkAgent(object):
    # simulated_world = None
    real_world = None
    model_file = "../../Resources/nn-data/model.json"
    weights_file = "../../Resources/nn-data/weights.h5"
    train_parameters = "../../Resources/train_parameters.json"

    data_storage = "../../Experiment_Data/"

    experienceBuffer = None
    lock_buffer = threading.Lock()

    working_collectors = 0

    vis = None

    pause = True
    stop = False

    def __init__(self):

        project_root_folder = os.path.join(os.getcwd(), "..", "..", "")

        # use the real world
        self.world = World()

        with open(self.train_parameters) as json_data:
            config = json.load(json_data)

        self.exploration_probability_start = config["exploration_probability_start"]
        self.exploration_probability_end = config["exploration_probability_end"]
        self.exploration_probability_runs_until_end = config["exploration_probability_runs_until_end"]
        self.exploration_probability_boost_runs_until_end = config["exploration_probability_boost_runs_until_end"]
        self.runs_before_update = config["runs_before_update"]
        self.runs_before_advancing_start = config["runs_before_advancing_start"]
        self.runs_before_testing_from_start = config["runs_before_testing_from_start"]

        self.exploration_probability = self.exploration_probability_start

        self.exploration_update_factor = (self.exploration_probability_end / self.exploration_probability_start) ** (
            1 / self.exploration_probability_runs_until_end)

        self.model = Factory.model_dueling_keras(self.world.STATE_DIMENSIONS, self.world.ACTIONS)

        # if there's already a model, use it. Else create new model
        if config["use_old_model"] and os.path.isfile(self.weights_file):
            # with open(self.model_file) as file:
            #    model_config = file.readline()
            #    self.model = model_from_json(model_config)
            self.model.load_weights(self.weights_file)

        else:
            # todo: check if keras allows saving of lambda layers
            # with open(self.model_file, 'w') as file:
            #    file.write(self.model.to_json())
            self.model.save_weights(self.weights_file)

        self.trainer = GPU_Trainer(project_root_folder)

        self.trainer.send_model_to_gpu()

        self.vis = TrainingVisualization(self.world.STATE_DIMENSIONS, self.world.ACTIONS, self.boost_exploration,
                                         self.change_mode, self.stop_training)

        self.experiment_directory = os.path.join(self.data_storage, time.strftime('%Y%m%d-%H%M%S'))
        os.makedirs(self.experiment_directory)

    def train(self):

        states = []
        actions = []
        rewards = []
        suc_states = []
        terminals = []

        statistics = {}
        statistics["steps"] = []
        statistics["rewards"] = []
        statistics["exploration_probability"] = []
        statistics["advancing_steps"] = []
        statistics["advancing_rewards"] = []
        statistics["testing_steps"] = []
        statistics["testing_rewards"] = []

        # statistics
        run_number = 1

        self.pause = True
        self.stop = False

        while True:

            while self.pause:
                if self.stop:
                    break
                self.vis.draw()
                time.sleep(.5)

            if self.stop:
                break

            new_states, new_actions, new_rewards, new_suc_states, new_terminals, run_steps, run_reward = self.run_until_terminal(
                self.exploration_probability)

            if run_steps:
                [states.append(s) for s in new_states]
                [actions.append(a) for a in new_actions]
                [rewards.append(r) for r in new_rewards]
                [suc_states.append(s) for s in new_suc_states]
                [terminals.append(t) for t in new_terminals]

                print("Run", run_number)

                # plot
                self.vis.update_step_graph(run_number, run_steps)
                self.vis.update_reward_graph(run_number, run_reward)
                self.vis.update_exploration_graph(run_number, self.exploration_probability)
                self.vis.draw()

                # collect statistics
                statistics["steps"].append(run_steps)
                statistics["rewards"].append(run_reward)
                statistics["exploration_probability"].append(self.exploration_probability)

                # train neural network after collecting some experience
                if run_number % self.runs_before_update == 0:
                    logging.debug("train neural net")

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

                # update exploration probability
                self.exploration_probability = max(
                    self.exploration_probability * self.exploration_update_factor,
                    self.exploration_probability_end)

                # try how far we get with the current model.
                # take last stable state as new starting point
                if run_number % self.runs_before_advancing_start == 0:
                    print("Advancing Start Position!")
                    self.world.last_action = None

                    new_states, new_actions, new_rewards, new_suc_states, new_terminals, run_steps, run_reward = self.run_until_terminal(
                        0)

                    if run_steps:
                        [states.append(s) for s in new_states]
                        [actions.append(a) for a in new_actions]
                        [rewards.append(r) for r in new_rewards]
                        [suc_states.append(s) for s in new_suc_states]
                        [terminals.append(t) for t in new_terminals]

                        # collect statistics
                        statistics["advancing_steps"].append(run_steps)
                        statistics["advancing_rewards"].append(run_reward)

                    #self.world.activate()
                    if self.world.is_at_goal():
                        self.world.reset_current_start_pose()
                    else:
                        # ideally, this only reverses the last move and restores the last non-terminal state
                        # if not, we're back to the old start pose
                        self.world.init_nonterminal_state()
                        self.world.update_current_start_pose()
                    #self.world.deactivate()
                    print("Advenced start by", run_steps, "steps")

                # reset to start pose and see how far we get
                # if we don't get better at this, we might consider measures like resetting to the last network
                # that worked best or boosting exploration
                if run_number % self.runs_before_testing_from_start == 0:

                    # reset to start
                    self.world.reset_current_start_pose()

                    new_states, new_actions, new_rewards, new_suc_states, new_terminals, run_steps, run_reward = self.run_until_terminal(
                        0)

                    if run_steps:
                        [states.append(s) for s in new_states]
                        [actions.append(a) for a in new_actions]
                        [rewards.append(r) for r in new_rewards]
                        [suc_states.append(s) for s in new_suc_states]
                        [terminals.append(t) for t in new_terminals]

                        statistics["testing_steps"].append(run_steps)
                        statistics["testing_rewards"].append(run_reward)

                        model_filename = os.path.join(self.experiment_directory, 'model-test-' + str(run_number))
                        self.model.save_weights(model_filename)

                    #self.world.activate()
                    if self.world.is_at_goal():
                        # once the wire is learned successfully, stop training
                        self.world.reset_current_start_pose()
                        print("Successfully completed wire!")
                        self.stop_training()
                    else:
                        # ideally, this only reverses the last move and restores the last nonterminal state
                        # if not, we're back to the old start pose
                        self.world.init_nonterminal_state()
                        self.world.update_current_start_pose()

                    #self.world.deactivate()

                    self.vis.update_test_step_graph(run_number, run_steps)

                    print(run_steps, "steps from start position")

                run_number += 1

        print("Stop training")

        statistics_file = os.path.join(self.experiment_directory, (time.strftime('%Y%m%d-%H%M%S') + '.csv'))

        cols = max([len(vals) for vals in statistics.values()])

        for key in statistics.keys():
            [statistics[key].append('') for _ in range(len(statistics[key]), cols)]

        with open(statistics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(statistics.keys())
            writer.writerows(zip(*statistics.values()))

        self.vis.show()

    def stop_training(self):
        self.stop = True

    def change_mode(self):
        self.pause = not self.pause

    def boost_exploration(self):
        self.exploration_probability = self.exploration_probability_start

        self.exploration_update_factor = (self.exploration_probability_end / self.exploration_probability_end) ** (
            1 / self.exploration_probability_boost_runs_until_end)

    def run_until_terminal(self, exploration_probability):

        states = []
        actions = []
        rewards = []
        suc_states = []
        terminals = []

        # statistics
        run_reward = 0
        run_steps = 0

        #self.world.activate()

        self.world.init_nonterminal_state()

        state, terminal = self.world.observe_state(), self.world.in_terminal_state()

        while not terminal:

            q_values = self.model.predict(np.expand_dims(state, axis=0))

            if np.any(np.isnan(q_values)):
                raise ValueError("Net is broken!")

            # explore with exploration_probability, else exploit
            if np.random.random() < exploration_probability:
                action = np.random.randint(self.world.ACTIONS)
            else:
                action = np.argmax(q_values)

            if action == np.argmax(q_values):
                self.vis.update_qvalue_graph(q_values, action, 'b', 'g')
            else:
                self.vis.update_qvalue_graph(q_values, action, 'b', 'r')

            self.vis.update_state_view(state)
            self.vis.draw()

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
                if run_steps:
                    # correct collected experience, since last action led to terminal state
                    rewards[-1] = e.args[1]
                    terminals[-1] = True

                    # replace last reward with reward proposed by exception
                    run_reward -= reward
                    run_reward += e.args[1]
                break
            except InsufficientProgressError:
                print("Insufficient Progress. Aborting Run.")
                break

        #self.world.deactivate()

        return states, actions, rewards, suc_states, terminals, run_steps, run_reward


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.DEBUG)

    agent = DeepQNetworkAgent()

    agent.train()
