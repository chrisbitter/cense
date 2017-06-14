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
from Cense.Visualization.visualization import training_visualization
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

    experienceBuffer = None
    lock_buffer = threading.Lock()

    working_collectors = 0

    vis = None

    def __init__(self):

        project_root_folder = os.path.join(os.getcwd(), "..", "..", "")

        # use the real world
        self.world = World()

        with open(self.train_parameters) as json_data:
            config = json.load(json_data)

        self.model = Factory.model_dueling_keras(self.world.STATE_DIMENSIONS, self.world.ACTIONS)

        # if there's already a model, use it. Else create new model
        if config["use_old_model"] and os.path.isfile(self.weights_file):
            # with open(self.model_file) as file:
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

        self.vis = training_visualization(self.world.STATE_DIMENSIONS, self.world.ACTIONS)

    def train(self):

        with open(self.train_parameters) as json_data:
            config = json.load(json_data)

        exploration_probability = config["exploration_probability_start"]
        runs_before_update = config["runs_before_update"]

        print("Train with exploration probability ", exploration_probability, "and updates after", runs_before_update,
              "runs")


        # statistics
        run_number = 1

        while True:

            # start new run
            with open(self.train_parameters) as json_data:
                config = json.load(json_data)

            if not config["do_train"]:
                break

            new_states, new_actions, new_rewards, new_suc_states, new_terminals, run_steps, run_reward = self.run_until_terminal(exploration_probability)

            if run_steps:
                [states.append(s) for s in new_states]
                [actions.append(a) for a in new_actions]
                [rewards.append(r) for r in new_rewards]
                [suc_states.append(s) for s in new_suc_states]
                [terminals.append(t) for t in new_terminals]

                print("Run: ", run_number, "\n\t", "steps:", run_steps, "\n\t", "reward:", run_reward)

                # plot statistics
                self.vis.update_step_graph(run_number, run_steps)
                self.vis.update_reward_graph(run_number, run_reward)
                self.vis.update_exploration_graph()
                self.vis.draw()

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

                # try how far we get with the current model.
                # take last stable state as new starting point
                if run_number % config["runs_before_advancing_start"] == 0:
                    print("Try advancing Start Position!")
                    self.world.reset()

                    new_states, new_actions, new_rewards, new_suc_states, new_terminals, run_steps, run_reward = self.run_until_terminal(0)

                    if run_steps:
                        [states.append(s) for s in new_states]
                        [actions.append(a) for a in new_actions]
                        [rewards.append(r) for r in new_rewards]
                        [suc_states.append(s) for s in new_suc_states]
                        [terminals.append(t) for t in new_terminals]

                    if self.world.is_at_goal():
                        self.world.reset_current_start_pose()
                    else:
                        # ideally, this only reverses the last move and restores the last nonterminal state
                        # if not, we're back to the old start pose
                        self.world.init_nonterminal_state()
                        self.world.update_current_start_pose()

                    print("\tAchieved reward", run_reward, "in", run_steps, "steps!")

                # reset to start pose and see how far we get
                # if we don't get better at this, we might consider measures like resetting to the last network
                # that worked best or boosting exploration
                if run_number % config["runs_before_testing_from_start"] == 0:
                    print("Try from Start Position!")

                    # reset to start
                    self.world.reset_current_start_pose()
                    self.world.reset()

                    new_states, new_actions, new_rewards, new_suc_states, new_terminals, run_steps, run_reward = self.run_until_terminal(0)

                    if run_steps:
                        [states.append(s) for s in new_states]
                        [actions.append(a) for a in new_actions]
                        [rewards.append(r) for r in new_rewards]
                        [suc_states.append(s) for s in new_suc_states]
                        [terminals.append(t) for t in new_terminals]

                    if self.world.is_at_goal():
                        self.world.reset_current_start_pose()
                    else:
                        # ideally, this only reverses the last move and restores the last nonterminal state
                        # if not, we're back to the old start pose
                        self.world.init_nonterminal_state()
                        self.world.update_current_start_pose()

                    print("\tAchieved reward", run_reward, "in", run_steps, "steps!")

                run_number += 1

        print("Stop training")

        statistics = np.dstack((vis.get_steps(), vis.get_rewards()))

        statistics = statistics.reshape(statistics.shape[1:])

        np.savetxt(time.strftime("%Y%m%d-%H%M%S") + ".csv", statistics, header="steps,reward")

        plt.show()

    def run_until_terminal(self, exploration_probability):

        states = []
        actions = []
        rewards = []
        suc_states = []
        terminals = []

        # statistics
        run_reward = 0
        run_steps = 0

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

        return states, actions, rewards, suc_states, terminals, run_steps, run_reward


if __name__ == '__main__':
    print("Starting from dqnAgent")
    agent = DeepQNetworkAgent()

    agent.train()
