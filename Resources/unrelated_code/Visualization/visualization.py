import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

class TrainingVisualization():
    def __init__(self, state_dimensions, num_actions, boost_exploration_callback, stop_callback):
        plt.figure(0)

        plots_total = 230
        plot_nr = 1
        # steps per run
        plt.subplot(plots_total + plot_nr)
        plt.title('steps')
        self.steps_plot, = plt.plot([], [])
        self.steps_ax = plt.gca()

        plot_nr += 1
        # plt.subplot(plots_total + plot_nr)
        # plt.title('reward')
        # self.rewards_plot, = plt.plot([], [])
        # self.rewards_ax = plt.gca()

        plot_nr += 1
        plt.subplot(plots_total + plot_nr)
        plt.title('exploration probability')
        self.exploration_plot, = plt.plot([], [])
        self.exploration_ax = plt.gca()

        plot_nr += 1
        plt.subplot(plots_total + plot_nr)
        action_names = ['right', 'left', 'forward', 'rot_right', 'rot_left']
        plt.xticks(range(len(action_names)), action_names, rotation='vertical')
        plt.title('q-value')
        # plt.bar
        self.bar_plot = plt.bar(list(range(num_actions)), np.zeros(num_actions))
        self.bar_ax = plt.gca()

        plot_nr += 1
        plt.subplot(plots_total + plot_nr)
        plt.title('state')
        self.cam_view = plt.imshow(np.zeros(state_dimensions), cmap='gray')
        self.cam_view.norm.vmax = 1

        plot_nr += 1
        plt.subplot(plots_total + plot_nr)
        plt.title('steps_test')
        self.test_step_plot, = plt.plot([], [])
        self.test_step_ax = plt.gca()

        plt.tight_layout()

        plt.figure(1, (2, 2))

        amnt_buttons = 2
        button_nr = 0
        self.axstop = plt.axes([0.05, button_nr / amnt_buttons, 0.9, 1 / amnt_buttons])
        self.bstop = Button(self.axstop, "Stop")
        self.bstop.on_clicked(stop_callback)

        button_nr = 1
        self.axboost = plt.axes([0.05, button_nr / amnt_buttons, 0.9, 1 / amnt_buttons])
        self.bboost = Button(self.axboost, "Boost Exploration")
        self.bboost.on_clicked(boost_exploration_callback)

    def update_step_graph(self, run_number, run_steps):
        self.steps_plot.set_xdata(np.append(self.steps_plot.get_xdata(), [run_number]))
        self.steps_plot.set_ydata(np.append(self.steps_plot.get_ydata(), [run_steps]))
        self.steps_ax.relim()
        self.steps_ax.autoscale_view()

    # def update_reward_graph(self, run_number, run_reward):
    #     self.rewards_plot.set_xdata(np.append(self.rewards_plot.get_xdata(), [run_number]))
    #     self.rewards_plot.set_ydata(np.append(self.rewards_plot.get_ydata(), [run_reward]))
    #     self.rewards_ax.relim()
    #     self.rewards_ax.autoscale_view()

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

    def update_test_step_graph(self, run_number, run_steps):
        self.test_step_plot.set_xdata(np.append(self.test_step_plot.get_xdata(), [run_number]))
        self.test_step_plot.set_ydata(np.append(self.test_step_plot.get_ydata(), [run_steps]))
        self.test_step_ax.relim()
        self.test_step_ax.autoscale_view()

    def draw(self):
        # plt.tight_layout()
        plt.draw()
        plt.pause(.001)

    def show(self):
        plt.show()


def dummy_callback():
    pass


if __name__ == "__main__":
    vis = TrainingVisualization((40, 40), 5, dummy_callback, dummy_callback)
