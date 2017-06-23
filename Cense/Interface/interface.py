# embedding_in_qt5.py --- Simple Qt5 application embedding matplotlib canvases
#
# Copyright (C) 2005 Florent Rougon
#               2006 Darren Dale
#               2015 Jens H Nielsen
#
# This file is an example program for matplotlib. It may be used and
# modified with no restriction; raw copies as well as modified versions
# may be distributed without limitation.

from __future__ import unicode_literals
import sys
import time
import numpy as np
from threading import Thread

import matplotlib
matplotlib.use('Qt5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Make sure that we are using QT5
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QAction
from PyQt5 import QtCore, QtWidgets


progname = "CENSE Trainer"
progversion = "1.0"


class MplCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, label="Custom Canvas", width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.suptitle(label)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        self.x = []
        self.y = []

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        # for i in range(10):
        #     self.update_figure(i, np.random.random())

    def update_figure(self, x, y):
        self.x.append(x)
        self.y.append(y)

        # self.axes.cla()
        self.axes.plot(self.x, self.y, marker='o')

        self.draw()
        time.sleep(.01)


class BarCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, label="Custom Canvas", width=5, height=4, dpi=100, num_actions=5):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.suptitle(label)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        self.num_actions = num_actions
        self.rects = self.axes.bar(list(range(self.num_actions)), [0] * self.num_actions)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def update_figure(self, q_values, action, highlight_color):
        for rect, q_val in zip(self.rects, q_values):
            rect.set_height(q_val)
            rect.set_color('b')

        self.rects[action].set_color(highlight_color)
        self.axes.relim()
        self.axes.autoscale()
        self.draw()
        time.sleep(.01)


class PictureCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, label="Custom Canvas", width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.suptitle(label)
        self.axes = fig.add_subplot(111)
        self.axes.axis('off')

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def update_figure(self, picture):
        self.axes.matshow(picture, cmap='gray').norm.vmax = 1
        self.draw()
        time.sleep(.01)


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self, stop_callback, boost_callback):
        self.stop_callback = stop_callback
        self.boost_callback = boost_callback

        QtWidgets.QMainWindow.__init__(self)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("application main window")

        self.file_menu = QtWidgets.QMenu('&File', self)
        self.file_menu.addAction('&Quit', self.fileQuit,
                                 QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.menuBar().addMenu(self.file_menu)

        self.help_menu = QtWidgets.QMenu('&Help', self)
        self.menuBar().addSeparator()
        self.menuBar().addMenu(self.help_menu)

        self.help_menu.addAction('&About', self.about)

        self.stopAction = QAction(QIcon(), 'Stop', self)
        self.stopAction.triggered.connect(self.stop)

        self.boostExploration = QAction(QIcon(), 'Boost', self)
        self.boostExploration.triggered.connect(self.boost)

        self.toolbar = self.addToolBar('Toolbar')
        self.toolbar.addAction(self.stopAction)
        self.toolbar.addAction(self.boostExploration)

        self.main_widget = QtWidgets.QWidget(self)

        l = QtWidgets.QGridLayout(self.main_widget)
        self.steps_canvas = MplCanvas(self.main_widget, label='Steps', width=5, height=4, dpi=100)
        self.exploration_canvas = MplCanvas(self.main_widget, label='Exploration Probability', width=5, height=4,
                                            dpi=100)
        self.q_value_canvas = BarCanvas(self.main_widget, label='Q-Values', width=5, height=4,
                                        dpi=100)
        self.state_canvas = PictureCanvas(self.main_widget, label='State', width=5, height=4,
                                          dpi=100)
        self.test_steps_canvas = MplCanvas(self.main_widget, label='Test Run Steps', width=5, height=4, dpi=100)

        l.addWidget(self.steps_canvas, 1, 1)
        l.addWidget(self.exploration_canvas, 2, 3)
        l.addWidget(self.q_value_canvas, 1, 3)
        l.addWidget(self.state_canvas, 1, 2)
        l.addWidget(self.test_steps_canvas, 2, 1)

        self.main_widget.setFocus()
        self.setCentralWidget(self.main_widget)

        self.statusBar().showMessage("Ready!")

    def stop(self):
        self.stop_callback()

    def boost(self):
        self.boost_callback()

    def set_status(self, new_status):
        self.statusBar().showMessage(new_status)

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

    def about(self):
        QtWidgets.QMessageBox.about(self, "About",
                                    """CENSE Training Environment""")


class Interface(object):
    qApp = None
    aw = None

    def __init__(self, stop_callback, boost_callback):
        self.stop_callback = stop_callback
        self.boost_callback = boost_callback

    def run(self):
        self.qApp = QtWidgets.QApplication(sys.argv)
        self.aw = ApplicationWindow(self.stop_callback, self.boost_callback)
        self.aw.setWindowTitle("%s" % progname)
        self.aw.show()
        sys.exit(self.qApp.exec_())

    @staticmethod
    def check_app_status(f):
        def wrapper(*args):
            if args[0].aw is not None:
                return f(*args)

        return wrapper

    @check_app_status
    def update_steps(self, run_number, run_steps):
        self.aw.steps_canvas.update_figure(run_number, run_steps)

    @check_app_status
    def update_exploration(self, run_number, exploration_probability):
        self.aw.exploration_canvas.update_figure(run_number, exploration_probability)

    @check_app_status
    def update_q_value(self, q_values, action, highlight_color):
        self.aw.q_value_canvas.update_figure(q_values, action, highlight_color)

    @check_app_status
    def update_state(self, state):
        self.aw.state_canvas.update_figure(state)

    @check_app_status
    def update_test_steps(self, run_number, run_steps):
        self.aw.test_steps_canvas.update_figure(run_number, run_steps)

    @check_app_status
    def set_status(self, *args):
        status = ""
        for arg in args:
            status += str(arg)
        self.aw.set_status(status)


if __name__ == "__main__":

    vis = Interface(print, print)

    t = Thread(target=vis.run)
    t.setDaemon(True)
    t.start()

    for t in range(50):
        time.sleep(1)
        vis.update_steps(t, t)
        vis.update_state(np.random.rand(t, t))
        vis.update_exploration(t, 1 / (t + 1))
        vis.update_q_value(np.array([t, t % 2, t % 3, t ** 2, -t]), np.random.randint(5), 'r')
        if t % 10 == 0:
            vis.update_test_steps(t, np.random.random())
