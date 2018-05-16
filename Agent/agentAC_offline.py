import json
import os
import time
import numpy as np
import pyqtgraph as pg
import tensorflow as tf
from PyQt5.QtCore import pyqtSignal

import NeuralNetwork.nnFactory as Factory
from Agent.Noise.emerging_gaussian import emerging_gaussian as Noise

#todo remove exceptions
from Environment.Robot.rtdeController import SpawnedInTerminalStateError, ExitedInTerminalStateError
from Environment.continuousEnvironment import UntreatableStateError, InsufficientProgressError
from Simulation.simulatedEnvironment import simulationEnvironment as World, IllegalPoseException
from Interface.interface import RunningMode
import logging
import os.path as path

from keras import backend as K

# silence tf compile warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class RunningStatus:
    RUN = 0
    PAUSE = 1
    STOP = 2


class AgentActorCritic(pg.QtCore.QThread):
    running_status = RunningStatus.STOP

    status_signal = pyqtSignal(object)
    steps_signal = pyqtSignal(object)
    state_signal = pyqtSignal(object)
    exploration_signal = pyqtSignal(object)
    test_steps_signal = pyqtSignal(object)
    actions_signal = pyqtSignal(object)

    def __init__(self, project_root, running_mode):
        super(AgentActorCritic, self).__init__()

        self.running_mode = running_mode

        parameter_file_path = os.path.abspath(os.path.join(project_root, "Resources/parameters.json"))

        # read configuration parameters
        with open(parameter_file_path) as json_data:
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
        self.data_storage = os.path.abspath(os.path.join(project_root, collector_config["data_storage"]))

        self.exploration_probability = self.exploration_probability_start

        self.exploration_update_factor = (self.exploration_probability_end / self.exploration_probability_start) ** (
            1 / self.exploration_probability_runs_until_end)

        self.world = World(config["environment"])

        self.model_file = path.abspath(path.join(*[project_root, config["trainer"]["gpu_settings"]["local_data_root"],
                                                   config["trainer"]["gpu_settings"]["local_model"]]))

        # create directory to store everything (statistics, NN-data, etc.)
        self.experiment_directory = os.path.join(self.data_storage, time.strftime('%Y%m%d-%H%M%S'))
        os.makedirs(self.experiment_directory)

        # persist parameters used for training
        with open(os.path.join(self.experiment_directory, 'train_parameters.json'), 'w') as f:
            json.dump(config, f, sort_keys=True, indent=4)

        K.set_learning_phase(0)
        # create neural network
        self.actor = Factory.actor_network(self.world.STATE_DIMENSIONS)
        self.actor_target = Factory.actor_network(self.world.STATE_DIMENSIONS)
        self.critic = Factory.critic_network(self.world.STATE_DIMENSIONS)
        self.critic_target = Factory.critic_network(self.world.STATE_DIMENSIONS)

        with open(os.path.join(self.experiment_directory, 'model_architecture.json'), 'w') as f:
            json.dump(self.actor.to_json(), f, sort_keys=True, indent=4)

        # if there's already a model, use it. Else create new model
        if (collector_config["resume_training"] or self.running_mode == RunningMode.PLAY) and os.path.isfile(
                self.model_file):
            self.actor.load_weights(self.model_file)
        else:
            self.actor.save_weights(self.model_file)

        ####

        self.batch_size = 32
        self.discount_factor = .99
        self.target_update_rate = .0001

        self.action_gradient = tf.placeholder(tf.float32, self.actor.output_shape)
        params_grad = tf.gradients(self.actor.output, self.actor.trainable_weights, -self.action_gradient)
        grads = zip(params_grad, self.actor.trainable_weights)
        optimizer = tf.train.AdamOptimizer(0.000005)
        self.optimize = optimizer.apply_gradients(grads)

        self.sess = K.get_session()

        gradients = K.gradients(self.critic.outputs, self.critic.inputs[1])[0]  # gradient tensors
        self.get_gradients = K.function(inputs=self.critic.inputs, outputs=[gradients])
        
        ####
        self.actor._make_predict_function()
        self.critic._make_predict_function()
        self.actor_target._make_predict_function()
        self.critic_target._make_predict_function()
        self.graph = tf.get_default_graph()

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

        reached_goal = False

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

                reached_goal = False

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
                            self.actor.save(os.path.join(self.experiment_directory,
                                                         'actor_' + str(run_number + 1) + '_' +
                                                         str(run_steps) + '.h5'))

                        if self.world.is_at_goal():
                            # once the wire is learned successfully, stop training
                            self.actor.save(
                                os.path.join(self.experiment_directory, 'actor_' + str(run_number + 1) + "_"
                                             + str(run_steps) + '_goal.h5'))
                            self.status_signal.emit("Successfully completed wire!")

                            if not self.continue_after_success:
                                self.stop_training()

                            # Exceptions thrown here will be caught and result in aborting training
                            self.world.reset(hard_reset=True)

                            reached_goal = True

                        else:
                            self.world.update_current_start_pose()

                        self.test_steps_signal.emit([run_number + 1, run_steps])

                    else:
                        statistics["testing_steps"] = 0
                        statistics["testing_rewards"] = ""

                except IllegalPoseException:
                    pass

                except UntreatableStateError as e:
                    self.status_signal.emit("Something went wrong! Stopping Training")
                    print(type(e))
                    self.stop_training()

            # try how far we get with the current model.
            # take last stable state as new starting point
            if run_number % self.runs_before_advancing_start == 0 and run_number % self.runs_before_testing_from_start != 0 and self.running_status == RunningStatus.RUN:
                self.status_signal.emit("Advancing Start Position")

                reached_goal = False

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

                        reached_goal = True

                        # reset to initial pose to train end of wire again
                        self.world.reset(hard_reset=True)
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
                # and (self.trainer.training_number > 0 or len(states) >= self.trainer.batch_size_start):
                print("Update Network")

                experience_buffer = range(len(states))

                # sample a minibatch
                minibatch = np.random.choice(experience_buffer, size=self.batch_size)

                # inputs are the states
                batch_states = np.array([states[i] for i in minibatch])  # bx(sxs)
                batch_actions = np.array([actions[i] for i in minibatch])  # bxa
                batch_rewards = np.array([rewards[i] for i in minibatch])  # bx1
                batch_new_states = np.array([new_states[i] for i in minibatch])  # bx(sxs)
                batch_terminals = np.array([terminals[i] for i in minibatch]).astype('bool')  # bx1

                with self.graph.as_default():
                    K.set_learning_phase(1)
                    target_q_values = self.critic_target.predict_on_batch(
                        [batch_new_states, self.actor_target.predict(batch_new_states)])  # bx1

                    y = batch_rewards.copy()  # bx1

                    for k in range(self.batch_size):
                        if not batch_terminals[k]:
                            y[k] += self.discount_factor * target_q_values[k]

                    # train critic with actual action-state value
                    self.critic.train_on_batch([batch_states, batch_actions], y)
                    # get action which would have been chosen by actor
                    actions_for_gradients = self.actor.predict(batch_states)  # bxa

                    # get action gradients w.r.t. critic output
                    action_grads = self.get_gradients([batch_states, actions_for_gradients])[0]  # bxa

                    # use action gradients to improve action chosen by actor
                    self.sess.run(self.optimize, feed_dict={
                        self.actor.inputs[0]: batch_states,
                        self.action_gradient: action_grads
                    })

                    actor_weights = self.actor.get_weights()
                    actor_target_weights = self.actor_target.get_weights()
                    for i in range(len(actor_weights)):
                        actor_target_weights[i] = self.target_update_rate * actor_weights[i] + \
                                                  (1 - self.target_update_rate) * actor_target_weights[i]
                    self.actor_target.set_weights(actor_target_weights)

                    # update critic target
                    critic_weights = self.critic.get_weights()
                    critic_target_weights = self.critic_target.get_weights()
                    for i in range(len(critic_weights)):
                        critic_target_weights[i] = self.target_update_rate * critic_weights[i] + \
                                                   (1 - self.target_update_rate) * critic_target_weights[i]
                    self.critic_target.set_weights(critic_target_weights)

                    self.actor.save_weights(self.model_file)
                    K.set_learning_phase(0)
                    self.actor.load_weights(self.model_file)

                training_number += 1

            run_steps = 0

            if not reached_goal:
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

        while not terminal and self.running_status == RunningStatus.RUN:

            state = self.world.observe_state()

            self.state_signal.emit(state)

            with self.graph.as_default():
                # print(np.expand_dims(state, axis=0))
                self.actor._make_predict_function()
                K.set_learning_phase(0)

                action_original = np.reshape(self.actor.predict(np.expand_dims(state, axis=0)), self.world.ACTIONS)


            if np.any(np.isnan(action_original)):
                raise ValueError("Net is broken!")

            action = np.empty_like(action_original)

            action[0] = Noise(action_original[0], exploration_probability, 0, 1)
            action[1] = Noise(action_original[1], exploration_probability, -1, 1)
            action[2] = Noise(action_original[2], exploration_probability, -1, 1)

            self.actions_signal.emit([action_original, action])

            try:
                # if action_log is not None:
                #     with open(action_log, 'a') as f:
                #         f.write(",".join(list(action)))
                #         f.write("\n")

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
                print("run_steps:", run_steps)
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

        self.actor.load_weights(self.model_file)

        action_log = os.path.join(self.experiment_directory, 'action_log.csv')

        action_keys = ["Forward", "Left", "Left Rotation"]

        with open(action_log, 'a') as f:
            f.write(",".join(action_keys))
            f.write("\n")

        run_number = 1

        while self.running_status == RunningStatus.RUN:

            try:
                self.world.reset(hard_reset=True)
                _, _, _, _, _, run_steps, _ = self.run_until_terminal(0, action_log)

                self.status_signal.emit("Needed " + str(run_steps) + " Steps")
                self.test_steps_signal.emit([run_number, run_steps])

            except UntreatableStateError as e:
                self.status_signal.emit("Something went wrong! Shutting down")
                print(type(e))
                self.stop_training()

            run_number += 1


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.ERROR)

    pass
