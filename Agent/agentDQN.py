import json
import logging
import os
import time
from copy import deepcopy
from threading import Thread

import NeuralNetwork.nnFactory as Factory
import numpy as np
import pyqtgraph as pg
import tensorflow as tf
from Cense.Agent.Trainer.gpuTrainer import GpuTrainer as Trainer
from PyQt5.QtCore import pyqtSignal

from Environment import DiscreteEnvironment as World
from Interface import RunningMode

# silence tf compile warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class RunningStatus:
    RUN = 0
    PAUSE = 1
    STOP = 2


class DeepQNetworkAgent(pg.QtCore.QThread):
    running_status = RunningStatus.STOP

    status_signal = pyqtSignal(object)
    steps_signal = pyqtSignal(object)
    state_signal = pyqtSignal(object)
    exploration_signal = pyqtSignal(object)
    test_steps_signal = pyqtSignal(object)
    actions_signal = pyqtSignal(object)

    def __init__(self, parameter_file, running_mode):
        super(DeepQNetworkAgent, self).__init__()

        self.running_mode = running_mode

        # read configuration parameters
        with open(parameter_file) as json_data:
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
        self.continue_after_success = collector_config["continue_after_success"]
        self.data_storage = collector_config["data_storage"]
        self.exploration_probability = self.exploration_probability_start

        self.exploration_update_factor = (self.exploration_probability_end / self.exploration_probability_start) ** (
            1 / self.exploration_probability_runs_until_end)

        self.world = World(config["environment"])

        self.model_file = config["trainer"]["gpu_settings"]["local_data_root"] + \
                          config["trainer"]["gpu_settings"]["local_model"]

        # create directory to store everything (statistics, NN-data, etc.)
        self.experiment_directory = os.path.join(self.data_storage, time.strftime('%Y%m%d-%H%M%S'))
        os.makedirs(self.experiment_directory)

        # persist parameters used for training
        with open(os.path.join(self.experiment_directory, 'train_parameters.json'), 'w') as f:
            json.dump(config, f, sort_keys=True, indent=4)

        # create neural network
        self.model = Factory.model_dueling(self.world.STATE_DIMENSIONS, self.world.ACTIONS)

        with open(os.path.join(self.experiment_directory, 'model_architecture.json'), 'w') as f:
            json.dump(self.model.to_json(), f, sort_keys=True, indent=4)

        # if there's already a model, use it. Else create new model
        if (collector_config["resume_training"] or self.running_mode == RunningMode.PLAY) and os.path.isfile(
                self.model_file):
            self.model.load_weights(self.model_file)
        else:
            self.model.save_weights(self.model_file)

        self.graph = tf.get_default_graph()

        self.trainer = Trainer(config["trainer"])

        if not collector_config["resume_training"]:
            self.trainer.reset()
            self.trainer.send_model_to_gpu()

    def run(self):
        if self.running_mode == RunningMode.TRAIN:
            self.train()
        elif self.running_mode == RunningMode.PLAY:
            self.play()

    def train(self):

        states = []
        actions = []
        rewards = []
        new_states = []
        terminals = []

        # statistics
        run_number = 0
        training_number = 1

        best_test_steps = 1

        statistics_file = os.path.join(self.experiment_directory, 'statistics.csv')

        statistics_keys = ["run_number", "steps", "rewards", "exploration_probability", "advancing_steps",
                           "advancing_rewards", "testing_steps", "testing_rewards", "successful_network_update"]

        with open(statistics_file, 'a') as f:
            f.write(",".join(statistics_keys))
            f.write("\n")

        while self.running_status == RunningStatus.RUN:

            # reset statistics
            statistics = {}

            for key in statistics_keys:
                statistics[key] = ""

            statistics["run_number"] = run_number + 1

            self.status_signal.emit("Run " + str(run_number + 1))

            # reset to start pose and see how far we get
            # if we don't get better at this, we might consider measures like resetting to the last network
            # that worked best or boosting exploration
            if run_number % self.runs_before_testing_from_start == 0 and self.running_status == RunningStatus.RUN:

                self.status_signal.emit("Test from Start")

                try:
                    # reset to start
                    self.world.reset(hard_reset=True)

                    _states, _actions, _rewards, _new_states, _terminals, run_steps, run_reward = \
                        self.run_until_terminal(0)

                    if run_steps:
                        [states.append(s) for s in _states]
                        [actions.append(a) for a in _actions]
                        [rewards.append(r) for r in _rewards]
                        [new_states.append(s) for s in _new_states]
                        [terminals.append(t) for t in _terminals]

                        statistics["testing_steps"] = run_steps
                        statistics["testing_rewards"] = run_reward

                        if run_steps > best_test_steps and run_steps >= self.min_steps_before_model_save:
                            self.model.save_weights(os.path.join(self.experiment_directory,
                                                                 'weights_' + str(run_number + 1) + '_' +
                                                                 str(run_steps) + '.h5'))

                        if self.world.is_at_goal():
                            # once the wire is learned successfully, stop training
                            self.model.save_weights(
                                os.path.join(self.experiment_directory, 'weights_' + str(run_number + 1)
                                             + str(run_steps) + '_goal.h5'))
                            self.status_signal.emit("Successfully completed wire!")

                            if not self.continue_after_success:
                                self.stop_training()

                            # Exceptions thrown here will be caught and result in aborting training
                            self.world.reset(hard_reset=True)

                        else:
                            self.world.update_current_start_pose()

                        self.test_steps_signal.emit([run_number + 1, run_steps])

                    else:
                        statistics["testing_steps"] = 0
                        statistics["testing_rewards"] = ""

                except IllegalPoseException:
                    pass

                except UntreatableStateError as e:
                    self.status_signal("Something went wrong! Stopping Training")
                    print(type(e))
                    self.stop_training()

            # try how far we get with the current model.
            # take last stable state as new starting point
            if run_number % self.runs_before_advancing_start == 0 and self.running_status == RunningStatus.RUN:
                self.status_signal.emit("Advancing Start Position")

                run_steps = 0

                try:
                    _states, _actions, _rewards, _new_states, _terminals, run_steps, run_reward = \
                        self.run_until_terminal(0)

                    if run_steps:
                        [states.append(s) for s in _states]
                        [actions.append(a) for a in _actions]
                        [rewards.append(r) for r in _rewards]
                        [new_states.append(s) for s in _new_states]
                        [terminals.append(t) for t in _terminals]

                        # collect statistics
                        statistics["advancing_steps"] = run_steps
                        statistics["advancing_rewards"] = run_reward
                    else:
                        statistics["advancing_steps"] = 0
                        statistics["advancing_rewards"] = ""

                    if self.world.is_at_goal():

                        # reset to initial pose to train end of wire again
                        self.world.reset()
                    else:
                        self.world.update_current_start_pose()

                    self.status_signal.emit("Advanced start by " + str(run_steps) + " steps")

                except IllegalPoseException:
                    pass

                except UntreatableStateError as e:
                    self.status_signal.emit("Something went wrong! Stopping Training")
                    print(type(e))
                    self.stop_training()

            # train neural network after collecting some experience
            if run_number % self.runs_before_update == 0 and self.running_status == RunningStatus.RUN:
                if self.trainer.is_done_training() and len(states):
                    print("Update Network: Success")
                    statistics["successful_network_update"] = 1
                    logging.debug("Replace NN and start new training")

                    # print(np.amin(rewards), np.amax(rewards))

                    # deep copy experience
                    gpu_states = deepcopy(states)  # states.copy() #(np.array(states)[:, 0]).tolist()
                    gpu_actions = deepcopy(actions)  # .copy()
                    gpu_rewards = deepcopy(rewards)  # .copy()
                    gpu_new_states = deepcopy(new_states)  # .copy() #(np.array(new_states)[:, 0]).tolist()
                    gpu_terminals = deepcopy(terminals)  # .copy()

                    # print(np.shape(gpu_states), np.shape(gpu_new_states))

                    with self.graph.as_default():
                        self.model.load_weights(self.model_file)

                    Thread(target=self.trainer.train,
                           args=(gpu_states, gpu_actions, gpu_rewards, gpu_new_states, gpu_terminals)).start()

                    training_number += 1

                    # clear experience collection
                    states = []
                    actions = []
                    rewards = []
                    new_states = []
                    terminals = []
                else:
                    # self.interface.set_status("Update Network: Failed")
                    statistics["successful_network_update"] = 0

            run_steps = 0

            while not run_steps and self.running_status == RunningStatus.RUN:
                # try to record a run
                # this is needed, because if for some reason the run_number isn't advanced, the testing, training or
                #  advancing section might be run over and over

                try:
                    _states, _actions, _rewards, _new_states, _terminals, run_steps, run_reward = \
                        self.run_until_terminal(self.exploration_probability)

                    if run_steps:
                        [states.append(s) for s in _states]
                        [actions.append(a) for a in _actions]
                        [rewards.append(r) for r in _rewards]
                        [new_states.append(s) for s in _new_states]
                        [terminals.append(t) for t in _terminals]

                        # plot
                        self.steps_signal.emit([run_number + 1, run_steps])
                        self.exploration_signal.emit([run_number + 1, self.exploration_probability])

                        # collect statistics
                        statistics["steps"] = run_steps
                        statistics["rewards"] = run_reward
                        statistics["exploration_probability"] = self.exploration_probability

                        # update exploration probability
                        self.exploration_probability = max(
                            self.exploration_probability * self.exploration_update_factor,
                            self.exploration_probability_end)

                        run_number += 1

                except IllegalPoseException:
                    pass

                except UntreatableStateError as e:
                    self.status_signal.emit("Something went wrong! Stopping Training")
                    print(type(e))
                    self.stop_training()

            statistics_string = ""

            for key in statistics_keys:
                statistics_string += str(statistics[key]) + ","

            # remove last comma
            statistics_string = statistics_string[:-1]

            with open(statistics_file, 'a') as f:
                f.write(statistics_string)
                f.write("\n")

    def start_training(self):
        self.running_status = RunningStatus.RUN

    def stop_training(self):
        self.running_status = RunningStatus.STOP

    def boost_exploration(self):
        self.exploration_probability = self.exploration_probability_start

        self.exploration_update_factor = (self.exploration_probability_end / self.exploration_probability_start) ** (
            1 / self.exploration_probability_boost_runs_until_end)

    def run_until_terminal(self, exploration_probability, action_log=None):

        states = []
        actions = []
        rewards = []
        new_states = []
        terminals = []

        # statistics
        run_reward = 0
        run_steps = 0

        # assumes world to be in nonterminal state, as promised by RealWorld
        terminal = False

        while not terminal:

            state = self.world.observe_state()

            self.state_signal.emit(state)

            with self.graph.as_default():
                q_values = self.model.predict(np.expand_dims(state, axis=0))

            if np.any(np.isnan(q_values)):
                raise ValueError("Net is broken!")

            # explore with exploration_probability, else exploit
            if np.random.random() < exploration_probability:
                action = np.random.randint(self.world.ACTIONS)
            else:
                action = np.argmax(q_values)

            self.actions_signal.emit([q_values[0], action])

            try:
                if action_log is not None:
                    with open(action_log, 'a') as f:
                        f.write(",".join(map(str, q_values)))
                        f.write("\n")

                new_state, reward, terminal = self.world.execute(action)

            except SpawnedInTerminalStateError:
                # last action apperantly led to undetected nonterminal state
                # correct last experience
                if len(states):
                    run_reward -= rewards[-1]
                    rewards[-1] = self.world.PUNISHMENT_WIRE
                    run_reward += rewards[-1]
                    terminals[-1] = True
                break
            except ExitedInTerminalStateError:
                # something went wrong when trying to reset to nonterminal state.
                # this also means, that new_state is terminal

                # just copy state (new_state not needed for DQN when terminal anyways)
                new_state = state
                reward = self.world.PUNISHMENT_WIRE
                terminal = True
            except UntreatableStateError:
                raise
            except InsufficientProgressError:
                self.status_signal.emit("Insufficient Progress. Aborting Run.")
                if len(states):
                    run_reward -= rewards[-1]
                    rewards[-1] = self.world.PUNISHMENT_INSUFFICIENT_PROGRESS
                    run_reward += self.world.PUNISHMENT_INSUFFICIENT_PROGRESS

                break
            except IllegalPoseException:
                self.world.reset(hard_reset=True)
                raise

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            new_states.append(new_state)
            terminals.append(terminal)

            # collect stats
            run_steps += 1
            run_reward += reward

        return states, actions, rewards, new_states, terminals, run_steps, run_reward

    def play(self):

        # self.model.load_weights(self.model_file)

        action_log = os.path.join(self.experiment_directory, 'action_log.csv')

        action_keys = ["Left Rotation", "Left", "Forward", "Right", "Right Rotation"]

        with open(action_log, 'a') as f:
            f.write(",".join(action_keys))
            f.write("\n")

        while True:
            try:
                self.world.reset(hard_reset=True)
                _, _, _, _, _, run_steps, run_reward = self.run_until_terminal(0, action_log)

                print(run_steps, run_reward)

            except UntreatableStateError as e:
                self.status_signal.emit("Something went wrong! Shutting down")
                print(type(e))
                self.stop_training()


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.ERROR)

    pass
