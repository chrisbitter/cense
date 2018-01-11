import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
import pyqtgraph.console
from PyQt5.QtCore import pyqtSignal
from pyqtgraph import PlotItem, ImageItem, ViewBox
import sys
import time
from Resources.Noise.ou import OU
from Cense.Agent.NeuralNetworkFactory.nnFactory import lstm


def bezier(points, resolution):
    if points.shape[0] < 2:
        raise IOError("Not enough points")
    if resolution < 2:
        raise IOError("Resolution too small")

    bezier_points = np.empty((resolution,) + points.shape[1:])

    for t in range(resolution):
        bezier_points[t] = recursive_bezier(points, t/resolution)

    return bezier_points


def recursive_bezier(points, t):

    if points.shape[0] < 1:
        raise IOError("not enough points")

    if points.shape[0] == 1:
        return points
    else:
        return (1-t)*recursive_bezier(points[:-1], t) + t*recursive_bezier(points[1:], t)


class Window(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        view = QtWidgets.QWidget()
        self.setCentralWidget(view)
        self.show()

        bezier_plot_widget = pg.PlotWidget()

        bezier_plot_item = bezier_plot_widget.getPlotItem()
        bezier_plot_item.enableAutoRange()
        bezier_plot_item.hideAxis('left')
        bezier_plot_item.hideAxis('bottom')

        self.bezier_plot = pg.ScatterPlotItem()
        bezier_plot_item.addItem(self.bezier_plot)
        bezier_plot_item.setTitle("bezier")

        self.layout = QtWidgets.QVBoxLayout(self)
        view.setLayout(self.layout)
        self.layout.addWidget(bezier_plot_widget)

    def add(self, points, brush='w'):
        self.bezier_plot.addPoints([{'pos': points[i], 'brush': brush} for i in range(points.shape[0])])

    def clear(self):
        self.bezier_plot.clear()

class Model(object):
    def get(self, input):
        pass
        # return np.array([[0, 0], [.2, 0], [.4, 0], [.8, 0]])


class LSTM(Model):
    def __init__(self, dim_input, dim_output):
        super(LSTM, self).__init__()
        self.noise = OU()
        self.model = lstm(dim_input, dim_output)

    def get(self, state, exploration):

        output = self.model.predict(state)
        output = self.noise.noise(output, output, 1-exploration, exploration)
        return output

class Environment(object):

    def __init__(self, support_points, point_dimensions):
        self.points_true = np.random.random((support_points, point_dimensions)) * 2 - 1
        self.bezier_points_true = bezier(self.points_true, 100)

    def observe(self):
        return self.bezier_points_true

    def execute(self, action):

        dead = True

        for p in self.bezier_points_true:
            if np.linalg.norm(p-action) < .2:
                dead = False
                break

        if dead:
            return -1, dead
        else:
            return .25, dead


class Agent(object):
    def __init__(self):

        app = QtWidgets.QApplication(sys.argv)
        app.setApplicationName("Bezier")

        window = Window()
        window.show()

        window.setGeometry(1000, 0, 800, 800)

        self.input_dim = (100,2)
        self.output_dim = (4,)

        self.model = LSTM(self.input_dim, self.output_dim)

        self.position = np.zeros(self.output_dim)

        self.environment = Environment(self.output_dim, self.input_dim[0])

        trials = 100
        exploration = .6

        for t in range(trials):

            #action = np.array([bezier_points_true[50].tolist(), [0.45, 0, 0], [0.5, 1, 0], [0.8, 0.8, 0]])

            terminal = False

            point_id = 1

            state = self.environment.observe()

            action = self.model.get(state, exploration)

            trajectory = bezier(action, self.input_dim[0])

            while not terminal and point_id < self.input_dim[0]:

                reward, terminal = self.environment.execute(trajectory[point_id])

                window.clear()

                window.add(self.environment.bezier_points_true, 'r')

                window.add(bezier(action, self.input_dim[0]), 'b')

                pg.QtGui.QGuiApplication.processEvents()

                time.sleep(.5)

            time.sleep(2)

        sys.exit(app.exec_())

if __name__ == "__main__":

    Agent()