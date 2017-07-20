# -*- coding: utf-8 -*-
"""
Example demonstrating a variety of scatter plot features.
"""

from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
import pyqtgraph.console
from pyqtgraph import PlotItem, ImageItem, ViewBox
import pyqtgraph as pg
import numpy as np
import time

from threading import Thread


def check_interface_status(f):
    def wrapper(*args):
        if args[0].ready:
            return f(*args)

    return wrapper


class Interface():
    ready = False
    t = None
    running_status = 'ready'
    mode = 'train'

    def __init__(self, start_callback, stop_callback, boost_callback):
        self.start_callback = start_callback
        self.stop_callback = stop_callback
        self.boost_callback = boost_callback

        self.t = Thread(target=self.run)
        self.t.setDaemon(True)
        self.t.start()

    def run(self):
        app = QtGui.QApplication([])
        self.mw = QtGui.QMainWindow()
        self.mw.resize(1500, 800)
        view = QtWidgets.QWidget()  ## GraphicsView with GraphicsLayout inserted by default
        self.mw.setCentralWidget(view)
        self.mw.show()
        self.mw.setWindowTitle('CENSE Demonstrator')

        layout = QtWidgets.QGridLayout()

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

        # Velocity Plot

        velocity_widget = pg.PlotWidget()

        velocity_plot_item = velocity_widget.getPlotItem()

        velocity_plot_item.setRange(xRange=(-2,2), yRange=(-1,4))

        a1 = pg.ArrowItem(angle=90, tipAngle=30, baseAngle=20, headLen=40, tailLen=20, tailWidth=8, pen=None,brush='b')
        self.new_pose_arrow = pg.ArrowItem(angle=90, tipAngle=30, baseAngle=20, headLen=40, tailLen=20, tailWidth=8, pen=None, brush='r')

        self.arrow_rotation = 0

        velocity_plot_item.addItem(a1)
        velocity_plot_item.addItem(self.new_pose_arrow)

        # self.velocity_plot = pg.BarGraphItem(x=range(3), height=np.zeros(3), width=1, brush='b')
        #
        # action_names = ['forward', 'right', 'rot_right']
        # xdict = dict(enumerate(action_names))
        #
        # stringaxis = pg.AxisItem(orientation='bottom')
        # stringaxis.setTicks([xdict.items()])
        #
        # velocity_widget = pg.PlotWidget(axisItems={'bottom': stringaxis})
        #
        # velocity_plot_item = velocity_widget.getPlotItem()
        # #velocity_plot_item.enableAutoRange()
        # velocity_plot_item.addItem(self.velocity_plot)
        # velocity_plot_item.setTitle("Velocities")
        # velocity_plot_item.setYRange(-1,1)

        # Test Steps Plot

        test_steps_widget = pg.PlotWidget()
        test_steps_plot_item = test_steps_widget.getPlotItem()
        test_steps_plot_item.enableAutoRange()
        self.test_steps_curve = test_steps_plot_item.plot()
        test_steps_plot_item.setTitle("Steps / Test Run")

        # Buttons

        self.statusAction = QtWidgets.QAction(QtGui.QIcon(), 'Start', self.mw)
        self.statusAction.triggered.connect(self.changestatus)

        self.boostExploration = QtWidgets.QAction(QtGui.QIcon(), 'Boost', self.mw)
        self.boostExploration.triggered.connect(self.boost)

        self.modeAction = QtWidgets.QAction(QtGui.QIcon(), 'Switch to Play', self.mw)
        self.modeAction.triggered.connect(self.changemode)

        self.toolbar = self.mw.addToolBar('Toolbar')
        self.toolbar.addAction(self.statusAction)
        self.toolbar.addAction(self.boostExploration)
        self.toolbar.addAction(self.modeAction)

        # Text display
        # self.text_widget = QtWidgets.QPlainTextEdit()
        # self.text_widget.ensureCursorVisible()
        # self.text_widget.setReadOnly(True)
        # self.text_widget.setBackgroundVisible(False)

        self.text_widget = QtWidgets.QLabel()
        self.text_widget.setAlignment(QtCore.Qt.AlignCenter)

        #layout.addWidget(steps_widget, 1, 1)
        #layout.addWidget(state_plot_widget, 1, 2)
        layout.addWidget(exploration_widget, 1, 3)
        layout.addWidget(test_steps_widget, 2, 1)
        layout.addWidget(velocity_widget, 2, 2)
        layout.addWidget(self.text_widget, 2, 3)

        view.setLayout(layout)

        self.ready = True

        import sys
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            QtGui.QApplication.instance().exec_()

    def changestatus(self):
        if self.running_status == 'ready':
            self.statusAction.setText('Stop')
            self.running_status = 'running'
            self.modeAction.setDisabled(True)
            self.start_callback()
        elif self.running_status == 'running':
            self.statusAction.setText('Exit')
            self.running_status = 'stopped'
            # self.statusAction.setDisabled(True)
            self.stop_callback()
        elif self.running_status == 'stopped':
            self.running_status = 'exit'
            self.statusAction.setDisabled(True)

    def changemode(self):
        if self.mode == 'train':
            self.modeAction.setText('Switch to Train')
            self.mode = 'play'
        elif self.mode == 'play':
            self.modeAction.setText('Switch to Play')
            self.mode = 'train'

    def boost(self):
        self.boost_callback()

    @check_interface_status
    def update_steps(self, run_number, run_steps):
        pass

        # x, y = self.steps_curve.getData()
        # if x is not None and y is not None:
        #     x = np.append(x, run_number)
        #     y = np.append(y, run_steps)
        #
        #     self.steps_curve.setData(x=x, y=y)
        # else:
        #     self.steps_curve.setData(x=[run_number], y=[run_steps])

    @check_interface_status
    def update_exploration(self, run_number, exploration_probability):

        #print("E-prob:", exploration_probability)

        x, y = self.exploration_curve.getData()
        if x is not None and y is not None:
            x = np.append(x, run_number)
            y = np.append(y, exploration_probability)

            self.exploration_curve.setData(x=x, y=y)
        else:
            self.exploration_curve.setData(x=[run_number], y=[exploration_probability])

    @check_interface_status
    def update_velocity(self, velocity):
        self.new_pose_arrow.setPos(velocity[1]*100, velocity[0]*100)

        self.new_pose_arrow.rotate(-self.arrow_rotation + velocity[2]*180/np.pi)
        self.arrow_rotation = velocity[2]*180/np.pi
        #self.new_pose_arrow.rotate(self.arrow_rotation)


    # @check_interface_status
    # def update_velocity(self, velocities, action):
    #     # draw velocitys except value corresponding to action
    #
    #     colors = ['b'] * 3
    #
    #     for i in range(3):
    #         if action[i] == 0:
    #             colors[i] = 'g'
    #         elif action[i] == 1:
    #             colors[i] = 'r'
    #
    #     self.velocity_plot.setOpts(height=velocities, brushes=colors)

    @check_interface_status
    def update_state(self, state):
        self.state_plot.setImage(np.rot90(state[0], 3))

    @check_interface_status
    def update_test_steps(self, run_number, run_steps):
        x, y = self.test_steps_curve.getData()
        if x is not None and y is not None:
            x = np.append(x, run_number)
            y = np.append(y, run_steps)

            self.test_steps_curve.setData(x=x, y=y)
        else:
            self.test_steps_curve.setData(x=[run_number], y=[run_steps])

    @check_interface_status
    def set_status(self, *args):
        status = ""
        for arg in args:
            status += str(arg)

        self.text_widget.setText(status)

        print(status)

        # self.text_widget.verticalScrollBar().setValue(self.text_widget.verticalScrollBar().maximum())
        # self.text_widget.appendPlainText(status)
        # self.text_widget.repaint()


go = False


def dummy_callback():
    global go
    go = not go


## Start Qt event loop unless running in interactive mode.
if __name__ == '__main__':
    interface = Interface(dummy_callback, dummy_callback, dummy_callback)

    while not interface.ready or not go:
        pass

    t = 1

    while go and t < 1000:
        now = time.time()
        interface.update_steps(t, np.random.random())
        interface.update_state([np.random.rand(40, 40), 0])

        if t % 10 == 0:
            interface.update_test_steps(t, np.random.random() * t)
        interface.update_velocity([t/10, t/10, 10*t*np.pi/180])
        interface.set_status(t)
        interface.update_exploration(t, time.time() - now)

        time.sleep(1)
        t += 1
