# -*- coding: utf-8 -*-
"""
Example demonstrating a variety of scatter plot features.
"""

from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
import pyqtgraph.console
from PyQt5.QtCore import pyqtSignal
from pyqtgraph import PlotItem, ImageItem, ViewBox
import pyqtgraph as pg
import numpy as np
import time
import json
import types
# from Cense.Agent.AcAgent import ActorCriticAgent as Agent

from threading import Thread
import sys
import os

sys.path.append("C:/Users/Cense/PycharmProjects/Cense/Cense")

class RunningMode:
    TRAIN = 1
    PLAY = 0


class CockpitWindow(QtWidgets.QMainWindow):
    def __init__(self, running_mode=None, parent=None):
        super(CockpitWindow, self).__init__(parent)

        view = QtWidgets.QWidget()
        self.setCentralWidget(view)
        self.show()

        self.running_mode = running_mode

        self.resize(1500, 800)

        title = "CENSE - Cognitive Enhanced Self-Optimization"

        self.action_plot = pg.BarGraphItem(x=range(6), height=np.zeros(6), width=1, brush='b')

        action_names = ['\u21a5', '\u21a4', '\u21b6', '\u21a5', '\u21a4', '\u21b6']
        xdict = dict(enumerate(action_names))

        font = QtGui.QFont()
        font.setPixelSize(40)
        stringaxis = pg.AxisItem(orientation='bottom')
        stringaxis.setTicks([xdict.items()])

        stringaxis.setStyle(tickTextOffset=10)
        stringaxis.tickFont = font
        stringaxis.setHeight(60)

        stringaxis.setLabel("Prediction | Execution")

        action_widget = pg.PlotWidget(axisItems={'bottom': stringaxis})

        action_plot_item = action_widget.getPlotItem()
        action_plot_item.enableAutoRange()
        action_plot_item.addItem(self.action_plot)
        action_plot_item.setTitle("Actions")

        action_plot_item.setYRange(-1, 1)

        from Agent.agentAC import AgentActorCritic as Agent

        project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))

        self.agent = Agent(project_root, self.running_mode)

        self.setWindowTitle(title)

        # Steps Plot

        steps_widget = pg.PlotWidget()
        steps_plot_item = steps_widget.getPlotItem()
        steps_plot_item.enableAutoRange()
        steps_plot_item.setDownsampling(mode='peak')
        self.steps_curve = steps_plot_item.plot()
        steps_plot_item.setTitle("Steps / Run")

        # State Plot

        state_plot_widget = pg.PlotWidget()

        state_plot_item = state_plot_widget.getPlotItem()
        state_plot_item.enableAutoRange()
        state_plot_item.hideAxis('left')
        state_plot_item.hideAxis('bottom')

        self.state_plot = ImageItem()
        state_plot_item.addItem(self.state_plot)
        state_plot_item.setTitle("State")

        # Exploration Probability Plot

        exploration_widget = pg.PlotWidget()
        exploration_plot_item = exploration_widget.getPlotItem()
        exploration_plot_item.enableAutoRange()
        self.exploration_curve = exploration_plot_item.plot()
        exploration_plot_item.setTitle("Exploration Probability")

        # Test Steps Plot

        test_steps_widget = pg.PlotWidget()
        test_steps_plot_item = test_steps_widget.getPlotItem()
        test_steps_plot_item.enableAutoRange()
        self.test_steps_curve = test_steps_plot_item.plot()
        test_steps_plot_item.setTitle("Test Runs")

        # Text display

        self.text_widget = QtWidgets.QLabel()
        self.text_widget.setAlignment(QtCore.Qt.AlignCenter)

        layout = QtWidgets.QGridLayout()
        view.setLayout(layout)

        # # layout.addWidget(steps_widget, 1, 1)
        # layout.addWidget(state_plot_widget, 1, 1)
        layout.addWidget(exploration_widget, 2, 2)
        # layout.addWidget(test_steps_widget, 2, 1)
        layout.addWidget(action_widget, 2, 1)
        # layout.addWidget(self.text_widget, 2, 2)

        layout.addWidget(state_plot_widget, 1, 1)
        layout.addWidget(test_steps_widget, 1, 2)

        self.agent.steps_signal.connect(self.update_steps)
        self.agent.state_signal.connect(self.update_state)
        self.agent.exploration_signal.connect(self.update_exploration)
        self.agent.test_steps_signal.connect(self.update_test_steps)
        self.agent.actions_signal.connect(self.update_actions)
        self.agent.status_signal.connect(self.set_status)

        # Buttons

        self.statusAction = QtWidgets.QAction(QtGui.QIcon(), 'Stop', self)
        self.statusAction.triggered.connect(self.agent.stop_training)

        self.boostExploration = QtWidgets.QAction(QtGui.QIcon(), 'Boost', self)
        self.boostExploration.triggered.connect(self.agent.boost_exploration)

        self.modeAction = QtWidgets.QAction(QtGui.QIcon(), 'Switch to Play', self)
        self.modeAction.triggered.connect(self.agent.start_training)
        self.modeAction.setDisabled(True)

        self.toolbar = self.addToolBar('Toolbar')
        self.toolbar.addAction(self.statusAction)
        # self.toolbar.addAction(self.boostExploration)
        # self.toolbar.addAction(self.modeAction)

        # self.setLayout(layout)

        # launch agent and gui
        # app.aboutToQuit.connect(self.agent.stop_training())
        self.agent.start_training()
        self.agent.start()

    def update_steps(self, data):
        x, y = self.steps_curve.getData()
        if x is not None and y is not None:
            x = np.append(x, data[0])
            y = np.append(y, data[1])

            self.steps_curve.setData(x=x, y=y)
        else:
            self.steps_curve.setData(x=[data[0]], y=[data[1]])

    def update_exploration(self, data):
        x, y = self.exploration_curve.getData()
        if x is not None and y is not None:
            x = np.append(x, data[0])
            y = np.append(y, data[1])

            self.exploration_curve.setData(x=x, y=y)
        else:
            self.exploration_curve.setData(x=[data[0]], y=[data[1]])

    def update_actions(self, data):
        # draw q_values except value corresponding to action

        colors = ['b'] * 3 + ['g'] * 3
        heights = []

        for i in range(3):
            heights.append(data[0][i])
        for i in range(3):
            heights.append(data[1][i])

        self.action_plot.setOpts(height=heights, brushes=colors)

    def update_state(self, data):
        self.state_plot.setImage(np.rot90(data, 3))

    def update_test_steps(self, data):
        x, y = self.test_steps_curve.getData()
        if x is not None and y is not None:
            x = np.append(x, data[0])
            y = np.append(y, data[1])

            self.test_steps_curve.setData(x=x, y=y)
        else:
            self.test_steps_curve.setData(x=[data[0]], y=[data[1]])

    def set_status(self, *args):
        status = ""
        for arg in args:
            status += str(arg)

        self.text_widget.setText(status)

        print(status)


class WindowRunningMode(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(WindowRunningMode, self).__init__(parent)

        self.description = QtWidgets.QLabel(self)
        self.description.setText("Choose Mode")

        self.pushButtonTRAIN = QtWidgets.QPushButton(self)
        self.pushButtonTRAIN.setText("Train")
        self.pushButtonTRAIN.clicked.connect(self.choose_train)

        self.pushButtonPLAY = QtWidgets.QPushButton(self)
        self.pushButtonPLAY.setText("Play")
        self.pushButtonPLAY.clicked.connect(self.choose_play)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.description)
        self.layout.addWidget(self.pushButtonTRAIN)
        self.layout.addWidget(self.pushButtonPLAY)

    @QtCore.pyqtSlot()
    def choose_train(self):
        self.close()
        self.cockpitWindow = CockpitWindow(RunningMode.TRAIN)
        self.cockpitWindow.show()

    @QtCore.pyqtSlot()
    def choose_play(self):
        self.close()
        self.cockpitWindow = CockpitWindow(RunningMode.PLAY)
        self.cockpitWindow.show()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("CENSE-Demonstrator")

    # window_running_mode = WindowRunningMode()
    # window_running_mode.show()

    cockpitWindow = CockpitWindow(RunningMode.TRAIN)
    cockpitWindow.show()

    sys.exit(app.exec_())
