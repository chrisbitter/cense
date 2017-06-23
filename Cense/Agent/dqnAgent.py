import logging

from Cense.World.Real.realWorld import RealWorld as World
from Cense.Trainer.gpuTrainer import GpuTrainer as Trainer
import Cense.NeuralNetworkFactory.nnFactory as Factory

from Cense.World.Real.realWorld import TerminalStateError, InsufficientProgressError

from Cense.Interface.interface import Interface as Interface

import os
import json
import csv
import time
import numpy as np
from threading import Thread

# silence tf compile warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class DeepQNetworkAgent(object):
    # simulated_world = None
    real_world = None
    model_file = "../../Resources/nn-data/model.json"
    weights_file = "../../Resources/nn-data/weights.h5"
    train_parameters = "../../Resources/train_parameters.json"

    data_storage = "../../Experiment_Data/"

    working_collectors = 0

    pause = True
    stop = False

    def __init__(self):

        project_root_folder = os.path.join(os.getcwd(), "..", "..", "")

        with open(self.train_parameters) as json_data:
            config = json.load(json_data)

        collector_config = config["collector"]

        self.exploration_probability_start = collector_config["exploration_probability_start"]
        self.exploration_probability_end = collector_config["exploration_probability_end"]
        self.exploration_probability_runs_until_end = collector_config["exploration_probability_runs_until_end"]
        self.exploration_probability_boost_runs_until_end = collector_config[
            "exploration_probability_boost_runs_until_end"]
        self.runs_before_update = collector_config["runs_before_update"]
        self.runs_before_advancing_start = collector_config["runs_before_advancing_start"]
        self.runs_before_testing_from_start = collector_config["runs_before_testing_from_start"]
        self.min_steps_before_model_save = collector_config["min_steps_before_model_save"]

        self.exploration_probability = self.exploration_probability_start

        self.exploration_update_factor = (self.exploration_probability_end / self.exploration_probability_start) ** (
            1 / self.exploration_probability_runs_until_end)

        self.interface = Interface(self.stop_training, self.boost_exploration)

        self.world = World(self.interface.set_status)

        self.model = Factory.model_dueling_keras(self.world.STATE_DIMENSIONS, self.world.ACTIONS)

        # if there's already a model, use it. Else create new model
        if collector_config["resume_training"] and os.path.isfile(self.weights_file):
            # with open(self.model_file) as file:
            #    model_config = file.readline()
            #    self.model = model_from_json(model_config)
            self.model.load_weights(self.weights_file)

        else:
            # todo: check if keras allows saving of lambda layers
            # with open(self.model_file, 'w') as file:
            #    file.write(self.model.to_json())
            self.model.save_weights(self.weights_file)

        self.trainer = Trainer(config["trainer"], self.interface.set_status)

        self.trainer.send_model_to_gpu()

        self.experiment_directory = os.path.join(self.data_storage, time.strftime('%Y%m%d-%H%M%S'))
        os.makedirs(self.experiment_directory)

    def train(self):

        states = []
        actions = []
        rewards = []
        suc_states = []
        terminals = []

        statistics = {
            "steps": [],
            "rewards": [],
            "exploration_probability": [],
            "advancing_steps": [],
            "advancing_rewards": [],
            "testing_steps": [],
            "testing_rewards": [],
            "successful_network_update": []
        }
        
        # statistics
        run_number = 1
        training_number = 1

        best_test_steps = 1

        self.stop = False

        while True:

            if self.stop:
                break

            try:
                new_states, new_actions, new_rewards, new_suc_states, new_terminals, run_steps, run_reward = \
                    self.run_until_terminal(self.exploration_probability)
            except TerminalStateError:
                break

            if run_steps:
                [states.append(s) for s in new_states]
                [actions.append(a) for a in new_actions]
                [rewards.append(r) for r in new_rewards]
                [suc_states.append(s) for s in new_suc_states]
                [terminals.append(t) for t in new_terminals]

                self.interface.set_status("Run ", str(run_number))

                # plot
                self.interface.update_steps(run_number, run_steps)
                self.interface.update_exploration(run_number, self.exploration_probability)

                # collect statistics
                statistics["steps"].append(run_steps)
                statistics["rewards"].append(run_reward)
                statistics["exploration_probability"].append(self.exploration_probability)

                # train neural network after collecting some experience
                if run_number % self.runs_before_update == 0:
                    if self.trainer.is_done_training():

                        statistics["successful_network_update"].append(True)
                        logging.debug("Replace NN and start new training")

                        self.model.load_weights(self.weights_file)
                        Thread(target=self.trainer.train,
                               args=(states, actions, rewards, suc_states, terminals)).start()

                        training_number += 1

                        # clear experience collection
                        states = []
                        actions = []
                        rewards = []
                        suc_states = []
                        terminals = []
                    else:
                        statistics["successful_network_update"].append(False)

                # update exploration probability
                self.exploration_probability = max(
                    self.exploration_probability * self.exploration_update_factor,
                    self.exploration_probability_end)

                # try how far we get with the current model.
                # take last stable state as new starting point
                if run_number % self.runs_before_advancing_start == 0:
                    self.interface.set_status("Advancing Start Position!")
                    self.world.last_action = None

                    try:
                        new_states, new_actions, new_rewards, new_suc_states, new_terminals, run_steps, run_reward = \
                            self.run_until_terminal(0)
                    except TerminalStateError:
                        break

                    if run_steps:
                        [states.append(s) for s in new_states]
                        [actions.append(a) for a in new_actions]
                        [rewards.append(r) for r in new_rewards]
                        [suc_states.append(s) for s in new_suc_states]
                        [terminals.append(t) for t in new_terminals]

                        # collect statistics
                        statistics["advancing_steps"].append(run_steps)
                        statistics["advancing_rewards"].append(run_reward)

                    if self.world.is_at_goal():
                        self.world.reset_current_start_pose()
                        self.world.reset()
                    else:
                        self.world.update_current_start_pose()
                    # self.world.deactivate()
                    self.interface.set_status("Advanced start by ", run_steps, " steps")

                # reset to start pose and see how far we get
                # if we don't get better at this, we might consider measures like resetting to the last network
                # that worked best or boosting exploration
                if run_number % self.runs_before_testing_from_start == 0:

                    # reset to start
                    self.world.reset_current_start_pose()

                    try:
                        new_states, new_actions, new_rewards, new_suc_states, new_terminals, run_steps, run_reward = \
                            self.run_until_terminal(0)
                    except TerminalStateError:
                        break

                    if run_steps:
                        [states.append(s) for s in new_states]
                        [actions.append(a) for a in new_actions]
                        [rewards.append(r) for r in new_rewards]
                        [suc_states.append(s) for s in new_suc_states]
                        [terminals.append(t) for t in new_terminals]

                        statistics["testing_steps"].append(run_steps)
                        statistics["testing_rewards"].append(run_reward)

                        if run_steps > best_test_steps and run_steps >= self.min_steps_before_model_save:
                            self.model.save_weights(self.model.save_weights(os.path.join(self.experiment_directory,
                                                                                         'weights_' + str(run_number)
                                                                                         + str(run_steps) + '.h5')))

                    if self.world.is_at_goal():
                        # once the wire is learned successfully, stop training
                        self.world.reset_current_start_pose()
                        self.world.reset()
                        self.model.save_weights(os.path.join(self.experiment_directory, 'weights_' + str(run_number)
                                                             + str(run_steps) + '_goal.h5'))
                        self.interface.set_status("Successfully completed wire!")
                        self.stop_training()
                    else:
                        self.world.update_current_start_pose()

                    self.interface.update_test_steps(run_number, run_steps)

                    self.interface.set_status(run_steps, "steps from start position")

                run_number += 1

        statistics_file = os.path.join(self.experiment_directory, 'statistics.csv')

        cols = max([len(vals) for vals in statistics.values()])

        for key in statistics.keys():
            [statistics[key].append('') for _ in range(len(statistics[key]), cols)]

        with open(statistics_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(statistics.keys())
            writer.writerows(zip(*statistics.values()))

    def stop_training(self):
        self.stop = True

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

        # assumes world to be in nonterminal state, as promised by RealWorld
        terminal = False

        state = self.world.observe_state()

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
                self.interface.update_q_value(q_values[0], action, 'g')
            else:
                self.interface.update_q_value(q_values[0], action, 'r')

            self.interface.update_state(state)

            try:
                suc_state, reward, terminal = self.world.execute(action)

                if suc_state is not None and reward is not None and terminal is not None:
                    states.append(state)
                    actions.append(action)
                    rewards.append(reward)
                    suc_states.append(suc_state)
                    terminals.append(terminal)

                    state = suc_state

                    # collect stats
                    run_steps += 1
                    run_reward += reward

            except TerminalStateError:
                # Abort everything
                self.interface.set_status("Robot stuck in Terminal State. Abort training.")
                self.stop_training()
                break
            except InsufficientProgressError:
                self.interface.set_status("Insufficient Progress. Aborting Run.")
                break

        # self.world.deactivate()

        return states, actions, rewards, suc_states, terminals, run_steps, run_reward


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.ERROR)

    agent = DeepQNetworkAgent()

    agent.train()
