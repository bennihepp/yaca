import sys, os, random
from PyQt4.QtCore import *
from PyQt4.QtGui import *

import numpy
import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import pyplot



class HistogramWindow(QWidget):

    PLOT_TYPE_NONE = 0
    PLOT_TYPE_BAR = 1
    PLOT_TYPE_HISTOGRAM = 2

    def __init__(self, parent=None):

        QWidget.__init__(self, parent)
        self.setWindowTitle('APC Histogram')

        self.plot_type = self.PLOT_TYPE_NONE

        self.build_widget()

        self.on_draw()


    def draw_barplot(self, values, x=None, x_labels=None):
        self.plot_type = self.PLOT_TYPE_BAR
        self.values = values
        self.x = x
        self.x_labels = x_labels
        self.on_draw()


    def draw_histogram(self, values, bins, bin_labels=None):
        self.plot_type = self.PLOT_TYPE_HISTOGRAM
        self.values = values
        self.bins = bins
        self.bin_labels = bin_labels
        self.on_draw()


    def on_draw(self):
        if self.plot_type == self.PLOT_TYPE_BAR:
            self.on_draw_barplot()
        elif self.plot_type == self.PLOT_TYPE_HISTOGRAM:
            self.on_draw_histogram()

    def on_draw_histogram(self):
        # Redraws the figure

        self.axes.clear()

        if self.values != None:

            if self.bin_labels != None:
                x = numpy.arange( self.bins )
                tmp = numpy.zeros( ( self.bins, ), int )
                for v in x:
                    tmp[ v ] += numpy.sum( self.values[ : ] == v )
                self.axes.bar( x, tmp, facecolor='yellow', alpha=0.75, align='center' )
                self.axes.set_xticks( x )
                self.axes.set_xticklabels( self.bin_labels )

            else:
                self.axes.hist( self.values, self.bins, facecolor='green', alpha=0.75, align='center' )

            self.axes.grid( True )

        self.canvas.draw()

    def on_draw_bar(self):
        # Redraws the figure

        self.axes.clear()

        if self.values != None:

            x = self.x
            if x == None:
                x = numpy.arange( self.values.shape[0] )

            self.axes.bar( x, values, facecolor='red', alpha=0.75, align='mid' )
            if self.x_labels != None:
                self.axes.set_xticks( x )
                self.axes.set_xticklabels( self.x_labels )

            self.axes.grid( True )

        self.canvas.draw()


    def build_widget(self):

        # Create the mpl Figure and FigCanvas objects. 
        # 5x4 inches, 100 dots-per-inch
        #
        self.dpi = 100
        #self.fig = Figure((5.0, 4.0), dpi=self.dpi)
        self.fig = pyplot.figure()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self)

        # Since we have only one plot, we can use add_axes 
        # instead of add_subplot, but then the subplot
        # configuration tool in the navigation toolbar wouldn't
        # work.
        #
        self.axes = self.fig.add_subplot(111)

        # Create the navigation toolbar, tied to the canvas
        #
        self.mpl_toolbar = NavigationToolbar(self.canvas, self)

        # Other GUI controls
        #

        vbox = QVBoxLayout()
        vbox.addWidget(self.canvas, 1)
        vbox.addWidget(self.mpl_toolbar)

        self.setLayout(vbox)

