import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

class training_visualization():
    def __init__(self, state_dimentions, num_actions):
        self.fig = plt.figure()
        # steps per run
        plt.subplot(331)
        plt.xlabel('run')
        plt.title('steps')
        self.steps_plot, = plt.plot([], [])
        self.steps_ax = plt.gca()

        plt.subplot(332)
        plt.xlabel('run')
        plt.title('reward')
        self.rewards_plot, = plt.plot([], [])
        self.rewards_ax = plt.gca()

        plt.subplot(333)
        plt.xlabel('action')
        plt.ylabel('q-value')
        # plt.bar
        self.bar_plot = plt.bar(list(range(num_actions)), np.zeros(num_actions))
        self.bar_ax = plt.gca()

        plt.subplot(334)
        plt.xlabel('run')
        plt.title('exploration probability')
        self.exploration_plot, = plt.plot([], [])
        self.exploration_ax = plt.gca()

        plt.subplot(335)
        self.cam_view = plt.imshow(np.zeros(state_dimentions), cmap='gray')
        self.cam_view.norm.vmax = 1

        plt.subplot(336)
        axprev = plt.axes([0.1, 0.05, 0.1, 0.075])
        axnext = plt.axes([0.1, 0.05, 0.1, 0.075])
        bnext = Button(axnext, 'Next')
        #bnext.on_clicked(callback.next)
        bprev = Button(axprev, 'Previous')
        #bprev.on_clicked(callback.prev)




    def update_step_graph(self, run_number, run_steps):
        self.steps_plot.set_xdata(np.append(self.steps_plot.get_xdata(), [run_number]))
        self.steps_plot.set_ydata(np.append(self.steps_plot.get_ydata(), [run_steps]))
        self.steps_ax.relim()
        self.steps_ax.autoscale_view()

    def update_reward_graph(self, run_number, run_reward):
        self.rewards_plot.set_xdata(np.append(self.rewards_plot.get_xdata(), [run_number]))
        self.rewards_plot.set_ydata(np.append(self.rewards_plot.get_ydata(), [run_reward]))
        self.rewards_ax.relim()
        self.rewards_ax.autoscale_view()
        pass

    def update_qvalue_graph(self, q_values, action, color, highlight_color):
        for rect, q_val in zip(self.bar_plot, q_values[0]):
            rect.set_height(q_val)
            rect.set_color(color)

        self.bar_plot[action].set_color(highlight_color)

        self.bar_ax.relim()
        self.bar_ax.autoscale_view()

    def update_state_view(self, state):
        self.cam_view.set_data(state)

    def update_exploration_graph(self, run_number, exploration_probability):
        self.exploration_plot.set_xdata(np.append(self.exploration_plot.get_xdata(), [run_number]))
        self.exploration_plot.set_ydata(np.append(self.exploration_plot.get_ydata(), [exploration_probability]))
        self.exploration_ax.relim()
        self.exploration_ax.autoscale_view()

    def draw(self):
        plt.tight_layout()
        plt.draw()
        plt.pause(.001)

    def get_steps(self):
        return self.steps_plot.get_ydata()

    def get_rewards(self):
        return self.rewards_plot.get_ydata()

if __name__ == "__main__":
    vis = training_visualization((40,40), 5)
    plt.tight_layout()
    plt.show()