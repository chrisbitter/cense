import csv
import json
import logging
import os
import time
from threading import Thread
import tensorflow as tf
import keras.models

import numpy as np

import Cense.Agent.NeuralNetworkFactory.nnFactory as Factory
# from Cense.Agent.Trainer.gpuTrainer import GpuTrainer as Trainer
from Cense.Agent.Trainer.gpuTrainer import GpuTrainer as Trainer
from Cense.Environment.continuousEnvironment import ContinuousEnvironment as World
from Cense.Environment.continuousEnvironment import *

# from Cense.Environment.dummyWorld import DummyWorld as World
# from Cense.Environment.dummyWorld import *

# silence tf compile warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class ActorCriticAgent(object):
    # simulated_world = None
    real_world = None
    project_root = "C:\\Users\\Christian\\Thesis\\workspace\\CENSE\\demonstrator_RLAlgorithm\\"

    # model_file = os.path.join(project_root, "Resources/nn-data/model.h5")
    # weights_file = os.path.join(project_root, "Resources/nn-data/weights.h5")
    # trained_weights_file = "../../Resources/nn-data/trained_weights.h5"
    train_parameters = os.path.join(project_root, "Resources/train_parameters_continuous.json")

    data_storage = os.path.join(project_root, "Experiment_Data/")

    start = False
    stop = False
    exit = True

    def __init__(self, set_status, update_steps, update_state, update_exploration, update_test_steps, update_actions):

        self.set_status = set_status
        self.update_steps = update_steps
        self.update_state = update_state
        self.update_exploration = update_exploration
        self.update_actions = update_actions

        self.update_test_steps = update_test_steps

        self.experiment_directory = os.path.join(self.data_storage, time.strftime('%Y%m%d-%H%M%S'))
        os.makedirs(self.experiment_directory)

        with open(self.train_parameters) as json_data:
            config = json.load(json_data)

        with open(os.path.join(self.experiment_directory, 'train_parameters.json'), 'w') as f:
            json.dump(config, f, sort_keys=True, indent=4)

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
        self.exploration_probability = self.exploration_probability_start

        self.exploration_update_factor = (self.exploration_probability_end / self.exploration_probability_start) ** (
            1 / self.exploration_probability_runs_until_end)

        self.world = World(config["environment"], self.set_status)

        self.model_file = config["trainer"]["gpu_settings"]["local_data_root"] + config["trainer"]["gpu_settings"][
            "local_model"]

        self.model = Factory.actor_network(self.world.STATE_DIMENSIONS)

        # if there's already a model, use it. Else create new model
        if collector_config["resume_training"] and os.path.isfile(self.model_file):
            self.model.load_weights(self.model_file)
            # self.model = keras.models.load_model(self.model_file)
            # self.model.load_weights(self.weights_file)
        else:
            # self.model = Factory.actor_network(self.world.STATE_DIMENSIONS)
            # self.model.save(self.model_file)
            self.model.save_weights(self.model_file)

        self.graph = tf.get_default_graph()

        self.trainer = Trainer(config["trainer"], self.set_status)

        self.trainer.reset()

        self.trainer.send_model_to_gpu()

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

        while True:

            if self.stop:
                break

            # reset statistics
            statistics = {}

            for key in statistics_keys:
                statistics[key] = ""

            statistics["run_number"] = run_number + 1

            self.set_status("Run ", str(run_number + 1))

            # reset to start pose and see how far we get
            # if we don't get better at this, we might consider measures like resetting to the last network
            # that worked best or boosting exploration
            if run_number % self.runs_before_testing_from_start == 0 and not self.stop:

                self.set_status("Test from Start")

                run_steps = 0

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
                            self.model.save(os.path.join(self.experiment_directory,
                                                         'actor_' + str(run_number + 1) + '_' +
                                                         str(run_steps) + '.h5'))

                        if self.world.is_at_goal():
                            # once the wire is learned successfully, stop training
                            self.model.save(
                                os.path.join(self.experiment_directory, 'actor_' + str(run_number + 1)
                                             + str(run_steps) + '_goal.h5'))
                            self.set_status("Successfully completed wire!")

                            if not self.continue_after_success:
                                self.stop_training()

                            # Exceptions thrown here will be caught and result in aborting training
                            self.world.reset(hard_reset=True)

                        else:
                            self.world.update_current_start_pose()

                        self.update_test_steps(run_number + 1, run_steps)

                    else:
                        statistics["testing_steps"] = 0
                        statistics["testing_rewards"] = ""


                except (UntreatableStateError, IllegalPoseException) as e:
                    self.set_status("Something went wrong! Stopping Training")
                    print(type(e))
                    self.stop_training()

            # try how far we get with the current model.
            # take last stable state as new starting point
            if run_number % self.runs_before_advancing_start == 0 and not self.stop:
                self.set_status("Advancing Start Position")

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

                    self.set_status("Advanced start by ", run_steps, " steps")

                except (UntreatableStateError, IllegalPoseException) as e:
                    self.set_status("Something went wrong! Stopping Training")
                    print(type(e))
                    self.stop_training()

            # train neural network after collecting some experience
            if run_number % self.runs_before_update == 0 and not self.stop:
                if self.trainer.is_done_training() and len(states):
                    self.set_status("Update Network: Success")
                    statistics["successful_network_update"] = 1
                    logging.debug("Replace NN and start new training")

                    # deep copy experience
                    gpu_states = np.array(states).tolist()
                    gpu_actions = actions.copy()
                    gpu_rewards = rewards.copy()
                    gpu_new_states = np.array(new_states).tolist()
                    gpu_terminals = terminals.copy()

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
                    # self.set_status("Update Network: Failed")
                    statistics["successful_network_update"] = 0

            run_steps = 0

            while not run_steps and not self.stop:
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
                        self.update_steps(run_number + 1, run_steps)
                        self.update_exploration(run_number + 1, self.exploration_probability)

                        # collect statistics
                        statistics["steps"] = run_steps
                        statistics["rewards"] = run_reward
                        statistics["exploration_probability"] = self.exploration_probability

                        # update exploration probability
                        self.exploration_probability = max(
                            self.exploration_probability * self.exploration_update_factor,
                            self.exploration_probability_end)

                        run_number += 1

                except (UntreatableStateError, IllegalPoseException) as e:
                    self.set_status("Something went wrong! Stopping Training")
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

        while self.exit:
            pass

    def start_training(self):
        self.start = True

    def stop_training(self):
        self.stop = True

    def exit_training(self):
        self.exit = True

    def boost_exploration(self):
        self.exploration_probability = self.exploration_probability_start

        self.exploration_update_factor = (self.exploration_probability_end / self.exploration_probability_start) ** (
            1 / self.exploration_probability_boost_runs_until_end)

    def run_until_terminal(self, exploration_probability):

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

            self.update_state(state)

            # print(self.model.summary())


            with self.graph.as_default():
                action_original = np.reshape(self.model.predict(np.expand_dims(state, axis=0)), self.world.ACTIONS)

            #print(action_original)

            if np.any(np.isnan(action_original)):
                raise ValueError("Net is broken!")


            # print(action_original.shape)

            # noise = np.empty_like(action_original)

            # for i in range(len(action_original)):
            #     # Ornstein-Uhlenbeck noise generation
            #     noise[i] = 0.15 * (0 - action_original[i]) + 0.2 * np.random.randn(1)

            # noise[0] = 0.15 * (.5 - action_original[0]) + 0.7 * np.random.randn(1) # forward
            # noise[1] = 0.15 * (0 - action_original[1]) + 0.7 * np.random.randn(1) # sideways
            # noise[2] = 0.15 * (0 - action_original[2]) + 0.7 * np.random.randn(1) # rotation

            #noise[0] = np.random.random() - .5
            #noise[1] = 3 * np.random.random() - 1.5
            #noise[2] = 3 * np.random.random() - 1.5


            # for i in range(len(action_original)):
            #     noise[i] = 2*np.random.random() - 1

            # noise *= exploration_probability

            # action = action_original + noise

            # randomly flip signs
            #if np.random.random() < exploration_probability:
            #    action[1] *= -1

            #if np.random.random() < exploration_probability:
            #    action[2] *= -1

            action = np.empty_like(action_original)

            # sample forward from gaussian with mean = action from actor and std depending on exploration
            while True:
                action[0] = np.random.normal(action_original[0], action_original[0]*exploration_probability)
                if 0 <= action[0] <= 1:
                    break

            # sample sideways from two gaussians with means = action, -action and std depending on exploration
            while True:
                if np.random.uniform() >= exploration_probability:
                    action[1] = np.random.normal(action_original[1], action_original[1]*exploration_probability)
                else:
                    action[1] = np.random.normal(-action_original[1], action_original[1]*exploration_probability)
                if -1 <= action[1] <= 1:
                    break

            # sample rotation from two gaussians with means = action, -action and std depending on exploration
            while True:
                if np.random.uniform() >= exploration_probability:
                    action[2] = np.random.normal(action_original[2],
                                                 action_original[2] * exploration_probability)
                else:
                    action[2] = np.random.normal(-action_original[2],
                                                 action_original[2] * exploration_probability)
                if -1 <= action[2] <= 1:
                    break

            self.update_actions(action_original, noise, action)

            try:
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
                self.set_status("Insufficient Progress. Aborting Run.")
                if len(states):
                    run_reward -= rewards[-1]
                    rewards[-1] = self.world.PUNISHMENT_INSUFFICIENT_PROGRESS
                    run_reward += self.world.PUNISHMENT_INSUFFICIENT_PROGRESS

                break
            except IllegalPoseException:
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

        self.model.load_weights(self.weights_file)

        while not self.stop:

            try:
                self.world.reset(hard_reset=True)

                self.run_until_terminal(0)

            except (UntreatableStateError, IllegalPoseException) as e:
                self.set_status("Something went wrong! Shutting down")
                print(type(e))
                self.stop_training()


run_numbers = np.zeros(3)
current_stats = np.zeros(3)

def print_stats(f):
    def wrapper(*args):
        if np.all(run_numbers == run_numbers[0]):
            print(current_stats)
        return f(*args)

    return wrapper


@print_stats
def update_steps(run_number, run_steps):
    run_numbers[0] = run_number
    current_stats[0] = run_steps


def update_state(state):
    pass


@print_stats
def update_exploration(run_number, exploration_probability):
    run_numbers[1] = run_number
    current_stats[1] = exploration_probability


@print_stats
def update_test_steps(run_number, run_steps):
    run_numbers[2] = run_number
    current_stats[2] = run_steps


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.ERROR)

    agent = ActorCriticAgent(print, update_steps, update_state, update_exploration, update_test_steps)

    agent.train()
