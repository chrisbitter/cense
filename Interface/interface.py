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


class Mode:
    DQN = 0
    AC = 1
    DUMMY = -1


class RunningMode:
    TRAIN = 0
    PLAY = 1


class CockpitWindow(QtWidgets.QMainWindow):
    def __init__(self, mode=None, running_mode=None, parent=None):
        super(CockpitWindow, self).__init__(parent)

        view = QtWidgets.QWidget()
        self.setCentralWidget(view)
        self.show()

        self.mode = mode
        self.running_mode = running_mode

        if self.mode == Mode.DQN:
            parameter_file = "C:\\Users\\Christian\\Thesis\\workspace\\CENSE\\demonstrator_RLAlgorithm\\Resources\\train_parameters_dqn.json"
        elif self.mode == Mode.AC:
            parameter_file = "C:\\Users\\Christian\\Thesis\\workspace\\CENSE\\demonstrator_RLAlgorithm\\Resources\\train_parameters_ac.json"
        elif self.mode == Mode.DUMMY:
            pass
        else:
            raise IOError("Unknown Mode")

        self.resize(1500, 800)

        title = "CENSE Demonstrator: "

        if self.mode == Mode.DQN:
            title += "DQN"

            # DQN-specific plots

            # Q-Value Plot
            self.action_plot = pg.BarGraphItem(x=range(5), height=np.zeros(5), width=1, brush='b')

            action_names = ['rotL', 'L', 'FW', 'R', 'rotR']
            xdict = dict(enumerate(action_names))

            stringaxis = pg.AxisItem(orientation='bottom')
            stringaxis.setTicks([xdict.items()])

            action_widget = pg.PlotWidget(axisItems={'bottom': stringaxis})

            action_plot_item = action_widget.getPlotItem()
            action_plot_item.enableAutoRange()
            action_plot_item.addItem(self.action_plot)
            action_plot_item.setTitle("Q-Values")

            from Agent.agentDQN import DeepQNetworkAgent as Agent
            parameter_file = "C:\\Users\\Christian\\Thesis\\workspace\\CENSE\\demonstrator_RLAlgorithm\\Resources\\train_parameters_dqn.json"

        elif self.mode == Mode.AC:
            title += "AC"

            # AC-specific plots

            self.action_plot = pg.BarGraphItem(x=range(6), height=np.zeros(6), width=1, brush='b')

            action_names = ['\u21a5', '\u21a4', '\u21b6','\u21a5', '\u21a4', '\u21b6']
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
            parameter_file = "C:\\Users\\Christian\\Thesis\\workspace\\CENSE\\demonstrator_RLAlgorithm\\Resources\\train_parameters_ac.json"

        elif self.mode == Mode.DUMMY:
            title += "Dummy"

            # AC-specific plots

            self.action_plot = pg.BarGraphItem(x=range(6), height=np.zeros(6), width=1, brush='b')

            action_names = ['\u21a5', '\u21a4', '\u21b6','\u21a5', '\u21a4', '\u21b6']
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

            Agent = DummyAgent
            parameter_file = ""
        else:
            raise IOError("Unknown mode!")

        self.agent = Agent(parameter_file, self.running_mode)

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

        #layout.addWidget(steps_widget, 1, 1)
        layout.addWidget(state_plot_widget, 1, 1)
        #layout.addWidget(exploration_widget, 1, 3)
        layout.addWidget(test_steps_widget, 2, 1)
        layout.addWidget(action_widget, 1, 2)
        layout.addWidget(self.text_widget, 2, 2)

        self.agent.steps_signal.connect(self.update_steps)
        self.agent.state_signal.connect(self.update_state)
        self.agent.exploration_signal.connect(self.update_exploration)
        self.agent.test_steps_signal.connect(self.update_test_steps)
        self.agent.actions_signal.connect(self.update_actions)
        self.agent.status_signal.connect(self.set_status)

        # Buttons

        self.statusAction = QtWidgets.QAction(QtGui.QIcon(), 'Stop Training', self)
        self.statusAction.triggered.connect(self.agent.stop_training)

        self.boostExploration = QtWidgets.QAction(QtGui.QIcon(), 'Boost', self)
        self.boostExploration.triggered.connect(self.agent.boost_exploration)

        self.modeAction = QtWidgets.QAction(QtGui.QIcon(), 'Switch to Play', self)
        self.modeAction.triggered.connect(self.agent.start_training)
        self.modeAction.setDisabled(True)

        self.toolbar = self.addToolBar('Toolbar')
        self.toolbar.addAction(self.statusAction)
        #self.toolbar.addAction(self.boostExploration)
        #self.toolbar.addAction(self.modeAction)

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

        if self.mode == Mode.DQN:
            colors = ['b'] * 5
            heights = data[0]

            if heights[data[1]] == np.amax(heights):
                colors[data[1]] = 'g'
            else:
                colors[data[1]] = 'r'

        elif self.mode == Mode.AC or self.mode == Mode.DUMMY:
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


class DummyAgent(pg.QtCore.QThread):
    status_signal = pyqtSignal(object)
    steps_signal = pyqtSignal(object)
    state_signal = pyqtSignal(object)
    exploration_signal = pyqtSignal(object)
    test_steps_signal = pyqtSignal(object)
    actions_signal = pyqtSignal(object)

    def __init__(self, parameter_file, _):
        super(DummyAgent, self).__init__()

    def run(self):
        for _ in range(100):
            i = np.random.randint(6)

            self.actions_signal.emit([np.random.rand(3).astype(np.float64), np.random.rand(3).astype(np.float64)])

            # if i == 0:
            #     self.status_signal.emit(str(_))
            # elif i == 1:
            #     self.steps_signal.emit([_, np.random.random()])
            # elif i == 2:
            #     self.state_signal.emit(np.random.random((10, 10)))
            # elif i == 3:
            #     self.exploration_signal.emit([_, np.random.random()])
            # elif i == 4:
            #     self.test_steps_signal.emit([_, np.random.random()])
            # else:
            #     self.actions_signal.emit(np.random.random((2, 6)))
            # time.sleep(.2)

    def boost_exploration(self):
        print("boost")

    def start_training(self):
        print("start")

    def stop_training(self):
        print("stop")


class WindowRunningMode(QtWidgets.QWidget):
    def __init__(self, mode=None, parent=None):
        super(WindowRunningMode, self).__init__(parent)

        self.mode = mode

        self.description = QtWidgets.QLabel(self)
        self.description.setText("Choose Mode")

        self.pushButtonTRAIN = QtWidgets.QPushButton(self)
        self.pushButtonTRAIN.setText("Train")
        self.pushButtonTRAIN.clicked.connect(self.on_pushButtonTRAIN_clicked)

        self.pushButtonPLAY = QtWidgets.QPushButton(self)
        self.pushButtonPLAY.setText("Play")
        self.pushButtonPLAY.clicked.connect(self.on_pushButtonPLAY_clicked)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.description)
        self.layout.addWidget(self.pushButtonTRAIN)
        self.layout.addWidget(self.pushButtonPLAY)

    @QtCore.pyqtSlot()
    def on_pushButtonTRAIN_clicked(self):
        self.close()
        self.cockpitWindow = CockpitWindow(self.mode, RunningMode.TRAIN)
        self.cockpitWindow.show()

    @QtCore.pyqtSlot()
    def on_pushButtonPLAY_clicked(self):
        self.close()
        self.cockpitWindow = CockpitWindow(self.mode, RunningMode.PLAY)
        self.cockpitWindow.show()

    @QtCore.pyqtSlot()
    def on_pushButtonDummy_clicked(self):
        self.close()
        self.parameterWindow = CockpitWindow(Mode.DUMMY)
        self.parameterWindow.show()


class WindowMode(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(WindowMode, self).__init__(parent)

        self.description = QtWidgets.QLabel(self)
        self.description.setText("Choose Learning Paradigm")

        self.pushButtonDQN = QtWidgets.QPushButton(self)
        self.pushButtonDQN.setText("Deep Q-Learning")
        self.pushButtonDQN.clicked.connect(self.on_pushButtonDQN_clicked)

        self.pushButtonAC = QtWidgets.QPushButton(self)
        self.pushButtonAC.setText("Actor Critic")
        self.pushButtonAC.clicked.connect(self.on_pushButtonAC_clicked)

        self.pushButtonDummy = QtWidgets.QPushButton(self)
        self.pushButtonDummy.setText("Dummy")
        self.pushButtonDummy.clicked.connect(self.on_pushButtonDummy_clicked)

        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.addWidget(self.description)
        self.layout.addWidget(self.pushButtonDQN)
        self.layout.addWidget(self.pushButtonAC)
        self.layout.addWidget(self.pushButtonDummy)

    @QtCore.pyqtSlot()
    def on_pushButtonDQN_clicked(self):
        self.close()
        self.runningModeWindow = WindowRunningMode(Mode.DQN)
        self.runningModeWindow.show()

    @QtCore.pyqtSlot()
    def on_pushButtonAC_clicked(self):
        self.close()
        self.runningModeWindow = WindowRunningMode(Mode.AC)
        self.runningModeWindow.show()

    @QtCore.pyqtSlot()
    def on_pushButtonDummy_clicked(self):
        self.close()
        self.runningModeWindow = WindowRunningMode(Mode.DUMMY)
        self.runningModeWindow.show()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("CENSE-Demonstrator")

    # window_cockpit = CockpitWindow(Mode.AC, RunningMode.TRAIN)
    # window_cockpit.show()

    window_running_mode = WindowRunningMode(Mode.AC)
    window_running_mode.show()

    # window_mode = WindowMode()
    # window_mode.show()

    sys.exit(app.exec_())
    #
    # # interface = Interface('dqn')
    # interface = Interface(Mode.DQN)
