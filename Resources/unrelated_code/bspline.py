import sys
import time

import numpy as np
import pyqtgraph as pg
from Agent.NeuralNetworkFactory import lstm
from pyqtgraph.Qt import QtWidgets

from Agent.Noise.ou import OU
from Resources.unrelated_code.bezier import bezier


def bspline(points, resolution):
    if points.shape[0] < 3:
        raise IOError("Not enough points")
    if resolution < 2:
        raise IOError("Resolution too small")


    midpoints = np.empty_like(points)

    midpoints[0] = points[0]
    midpoints[-1] = points[-1]

    for i in range(1, points.shape[0]-1):
        midpoints[i] = points[i-1]/6 + 2*points[i]/3 + points[i+1]/6

    curve_points = np.empty((points.shape[0]-1, resolution) + points.shape[1:])



    for i in range(1, points.shape[0]):
        curve_points[(i-1)*resolution:i*resolution] = bezier(np.stack((midpoints[i-1], 2*points[i-1]/3+points[i]/3,
                                                        points[i-1]/3+2*points[i]/3, midpoints[i])), resolution)

    print(curve_points.shape)

    return curve_points

class Window(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        view = QtWidgets.QWidget()
        self.setCentralWidget(view)
        self.show()

        bezier_plot_widget = pg.PlotWidget()

        bezier_plot_item = bezier_plot_widget.getPlotItem()
        bezier_plot_item.enableAutoRange()
        #bezier_plot_item.hideAxis('left')
        #bezier_plot_item.hideAxis('bottom')

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
    def get(self, input, exploration):
        pass
        # return np.array([[0, 0], [.2, 0], [.4, 0], [.8, 0]])


class LSTM(Model):
    def __init__(self, dim_input, dim_output):
        super(LSTM, self).__init__()
        self.noise = OU()
        self.model = lstm(dim_input, dim_output)

    def get(self, state, exploration):

        output = self.model.predict(np.expand_dims(state, axis=0))
        output = self.noise.noise(output, output, 1-exploration, exploration)
        return output

class Environment(object):

    def __init__(self, support_points, point_dimensions):
        self.points_true = np.random.random((support_points+1, point_dimensions)) * 2 - 1
        self.points_true[0] = np.zeros(point_dimensions)
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

        spline_resolution = 100
        spline_amount = 5
        dimension = 2

        app = QtWidgets.QApplication(sys.argv)
        app.setApplicationName("Bezier")

        window = Window()
        window.show()

        window.setGeometry(1000, 0, 800, 800)

        self.dim_input = (spline_resolution , dimension)
        self.dim_output = (spline_amount, dimension)

        self.model = LSTM(self.dim_input, self.dim_output)

        self.position = np.zeros(self.dim_output)

        self.environment = Environment(spline_amount, dimension)

        trials = 1
        exploration = .6

        for t in range(trials):

            #action = np.array([bezier_points_true[50].tolist(), [0.45, 0, 0], [0.5, 1, 0], [0.8, 0.8, 0]])

            terminal = False

            point_id = 1

            state = self.environment.observe()

            action = self.model.get(state, exploration)

            action = np.insert(action[0], 0, np.zeros(dimension))

            trajectory = bspline(action, spline_resolution)

            while not terminal and point_id < self.dim_input[0]:

                reward, terminal = self.environment.execute(trajectory[0, point_id])

                window.clear()

                window.add(self.environment.bezier_points_true, 'r')

                splines = bspline(np.reshape(action, self.dim_output), self.dim_input[0])

                for s in range(splines.shape[0]):

                    window.add(splines[s], 'b')

                pg.QtGui.QGuiApplication.processEvents()

                time.sleep(.5)

            time.sleep(2)

        sys.exit(app.exec_())

if __name__ == "__main__":

    Agent()