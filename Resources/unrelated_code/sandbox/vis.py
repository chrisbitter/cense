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
import os
import random
import matplotlib
import numpy as np

# Make sure that we are using QT5
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QAction

matplotlib.use('Qt5Agg')
from PyQt5 import QtCore, QtWidgets

from numpy import arange, sin, pi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

progname = os.path.basename(sys.argv[0])
progversion = "0.1"


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

        self.update_figure(5,5)

    def update_figure(self, x, y):
        self.x.append(x)
        self.y.append(y)

        self.axes.cla()
        self.axes.plot(self.x, self.y, 'b', marker='o')
        self.draw()


class BarCanvas(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""

    def __init__(self, parent=None, label="Custom Canvas", width=5, height=4, dpi=100, num_actions=5):
        fig = Figure(figsize=(width, height), dpi=dpi)
        fig.suptitle(label)
        self.axes = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        self.num_actions = num_actions
        self.rects = self.axes.bar(list(range(self.num_actions)), [1] * self.num_actions)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def update_figure(self, q_values, action, highlight_color):

        for rect, q_val in zip(self.rects, q_values):
            rect.set_height(q_val)
            rect.set_color('b')

        self.rects[action].set_color(highlight_color)
        self.draw()

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
        self.axes.cla()
        self.axes.matshow(picture, cmap='gray').norm.vmax = 1
        self.draw()


class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
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

        stopAction = QAction(QIcon(), 'Stop', self)
        #stopAction.setShortcut('Ctrl+Q')
        stopAction.triggered.connect(self.stop)

        boostExploration = QAction(QIcon(), 'Boost', self)
        boostExploration.triggered.connect(self.boost)

        self.toolbar = self.addToolBar('Exit')
        self.toolbar.addAction(stopAction)

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
        print("stop")

    def boost(self):
        pass

    def change_status(self, new_status):
        self.statusBar().showMessage(new_status)

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

    def about(self):
        QtWidgets.QMessageBox.about(self, "About",
                                    """embedding_in_qt5.py example
Copyright 2005 Florent Rougon, 2006 Darren Dale, 2015 Jens H Nielsen

This program is a simple example of a Qt5 application embedding matplotlib
canvases.

It may be used and modified with no restriction; raw copies as well as
modified versions may be distributed without limitation.

This is modified from the embedding in qt4 example to show the difference
between qt4 and qt5"""
                                    )


if __name__ == "__main__":
    qApp = QtWidgets.QApplication(sys.argv)

    aw = ApplicationWindow()
    aw.setWindowTitle("%s" % progname)
    aw.show()

    agent = DeepQNetworkAgent()

    agent.train()


    sys.exit(qApp.exec_())
