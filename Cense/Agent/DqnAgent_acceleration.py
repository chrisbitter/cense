import csv
import json
import logging
import os
import time
from  copy import deepcopy
from threading import Thread

import numpy as np

import Cense.Agent.NeuralNetworkFactory.nnFactory as Factory
from Cense.Agent.Trainer.gpuTrainer import GpuTrainer as Trainer
from Cense.Environment.realEnvironment_acceleration import RealEnvironment as World
from Cense.Environment.realEnvironment_acceleration import *
from Cense.Interface.pyqtgraphInterface_acceleration import Interface as Interface

# silence tf compile warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class DeepQNetworkAgent(object):
    # simulated_world = None
    real_world = None
    model_file = "../../Resources/nn-data/model.json"
    weights_file = "../../Resources/nn-data/weights.h5"
    # trained_weights_file = "../../Resources/nn-data/trained_weights.h5"
    train_parameters = "../../Resources/train_parameters_acceleration.json"

    data_storage = "../../Experiment_Data/"

    working_collectors = 0

    start = False
    stop = False

    def __init__(self):

        self.interface = Interface(self.start_training, self.stop_training, self.boost_exploration)

        while not self.start:
            pass

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

        self.world = World(config["environment"], self.interface.set_status)

        self.model = Factory.model_acceleration_q(self.world.STATE_DIMENSIONS, self.world.VELOCITY_DIMENSIONS, self.world.ACTIONS)

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

        self.trainer.reset()

        self.trainer.send_model_to_gpu()

    def train(self):

        states = []
        actions = []
        rewards = []
        suc_states = []
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

            self.interface.set_status("Run ", str(run_number + 1))

            # reset to start pose and see how far we get
            # if we don't get better at this, we might consider measures like resetting to the last network
            # that worked best or boosting exploration
            if run_number % self.runs_before_testing_from_start == 0 and not self.stop:

                self.interface.set_status("Test from Start")

                run_steps = 0

                try:
                    # reset to start
                    self.world.reset(hard_reset=True)

                    new_states, new_actions, new_rewards, new_suc_states, new_terminals, run_steps, run_reward = \
                        self.run_until_terminal(0)

                    if run_steps:
                        [states.append(s) for s in new_states]
                        [actions.append(a) for a in new_actions]
                        [rewards.append(r) for r in new_rewards]
                        [suc_states.append(s) for s in new_suc_states]
                        [terminals.append(t) for t in new_terminals]

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
                            self.interface.set_status("Successfully completed wire!")

                            if not self.continue_after_success:
                                self.stop_training()

                            # Exceptions thrown here will be caught and result in aborting training
                            self.world.reset(hard_reset=True)

                        else:
                            self.world.update_current_start_pose()

                        self.interface.update_test_steps(run_number + 1, run_steps)

                    else:
                        statistics["testing_steps"] = 0
                        statistics["testing_rewards"] = ""


                except (UntreatableStateError, IllegalPoseException) as e:
                    self.interface.set_status("Something went wrong! Stopping Training")
                    print(type(e))
                    self.stop_training()

            # try how far we get with the current model.
            # take last stable state as new starting point
            if run_number % self.runs_before_advancing_start == 0 and not self.stop:
                self.interface.set_status("Advancing Start Position")

                run_steps = 0

                try:
                    new_states, new_actions, new_rewards, new_suc_states, new_terminals, run_steps, run_reward = \
                        self.run_until_terminal(0)

                    if run_steps:
                        [states.append(s) for s in new_states]
                        [actions.append(a) for a in new_actions]
                        [rewards.append(r) for r in new_rewards]
                        [suc_states.append(s) for s in new_suc_states]
                        [terminals.append(t) for t in new_terminals]

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

                    self.interface.set_status("Advanced start by ", run_steps, " steps")

                except (UntreatableStateError, IllegalPoseException) as e:
                    self.interface.set_status("Something went wrong! Stopping Training")
                    print(type(e))
                    self.stop_training()

            # train neural network after collecting some experience
            if run_number % self.runs_before_update == 0 and not self.stop:
                if self.trainer.is_done_training() and len(states):
                    self.interface.set_status("Update Network: Success")
                    statistics["successful_network_update"] = 1
                    logging.debug("Replace NN and start new training")

                    self.model.load_weights(self.weights_file)

                    # deep copy experience
                    gpu_states = (np.array(states)[:,0]).tolist()
                    gpu_velocities = (np.array(states)[:,1]).tolist()
                    gpu_actions = actions.copy()
                    gpu_rewards = rewards.copy()
                    gpu_suc_states = (np.array(suc_states)[:, 0]).tolist()
                    gpu_suc_velocities = (np.array(suc_states)[:, 1]).tolist()
                    gpu_terminals = terminals.copy()

                    Thread(target=self.trainer.train,
                           args=(gpu_states, gpu_actions, gpu_rewards, gpu_suc_states, gpu_terminals, gpu_velocities, gpu_suc_velocities)).start()

                    training_number += 1

                    # clear experience buffer
                    states = []
                    actions = []
                    rewards = []
                    suc_states = []
                    terminals = []


                else:
                    self.interface.set_status("Update Network: Failed")
                    statistics["successful_network_update"] = 0

            run_steps = 0

            while not run_steps and not self.stop:
                # try to record a run
                # this is needed, because if for some reason the run_number isn't advanced, the testing, training or
                #  advancing section might be run over and over

                try:
                    new_states, new_actions, new_rewards, new_suc_states, new_terminals, run_steps, run_reward = \
                        self.run_until_terminal(self.exploration_probability)

                    if run_steps:
                        [states.append(s) for s in new_states]
                        [actions.append(a) for a in new_actions]
                        [rewards.append(r) for r in new_rewards]
                        [suc_states.append(s) for s in new_suc_states]
                        [terminals.append(t) for t in new_terminals]

                        # plot
                        self.interface.update_steps(run_number + 1, run_steps)
                        self.interface.update_exploration(run_number + 1, self.exploration_probability)

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
                    self.interface.set_status("Something went wrong! Stopping Training")
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

        while self.interface.running_status is not 'exit':
            pass


    def start_training(self):
        self.start = True

    def stop_training(self):
        self.stop = True

    def boost_exploration(self):
        self.exploration_probability = self.exploration_probability_start

        self.exploration_update_factor = (self.exploration_probability_end / self.exploration_probability_start) ** (
            1 / self.exploration_probability_boost_runs_until_end)

    def run_until_terminal(self, exploration_probability):

        logging.debug('run_until_terminal')

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

        while not terminal:

            state = self.world.observe_state()

            self.interface.update_state(state)

            q_values = self.model.predict([np.expand_dims(state[0], axis=0), np.expand_dims(state[1], axis=0)])

            if np.any(np.isnan(q_values)):
                raise ValueError("Net is broken!")

            # explore with exploration_probability, else exploit

            if np.random.random() < exploration_probability:
                # explore
                action = np.random.randint(self.world.ACTIONS[1], size=self.world.ACTIONS[0])
            else:
                # exploit
                action = [np.argmax(q_values[0][i]) for i in range(self.world.ACTIONS[0])]

            self.interface.update_velocity(self.world.get_normalized_velocities(), action)

            try:
                suc_state, reward, terminal = self.world.execute(action)

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
                # this also means, that suc_state is terminal

                # just copy state (suc_state not needed for DQN when terminal anyways)
                suc_state = state
                reward = self.world.PUNISHMENT_WIRE
                terminal = True
            except UntreatableStateError:
                raise
            except InsufficientProgressError:
                self.interface.set_status("Insufficient Progress. Aborting Run.")
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
            suc_states.append(suc_state)
            terminals.append(terminal)

            # collect stats
            run_steps += 1
            run_reward += reward

        logging.debug('run_until_terminal done')

        return states, actions, rewards, suc_states, terminals, run_steps, run_reward


    def play(self):

        self.model.load_weights(self.weights_file)

        while not self.stop:

            try:
                self.world.reset(hard_reset=True)

                new_states, new_actions, new_rewards, new_suc_states, new_terminals, run_steps, run_reward = \
                    self.run_until_terminal(0)

            except (UntreatableStateError, IllegalPoseException) as e:
                self.interface.set_status("Something went wrong! Shutting down")
                print(type(e))
                self.stop_training()


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.ERROR)

    agent = DeepQNetworkAgent()

    if agent.interface.mode == 'train':
        agent.train()
    elif agent.interface.mode == 'play':
        agent.play()
