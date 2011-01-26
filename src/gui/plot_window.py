import sys, os, random
from PyQt4.QtCore import *
from PyQt4.QtGui import *

import numpy
import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import pyplot



class PlotWindow(QWidget):

    PLOT_TYPE_NONE = 0
    PLOT_TYPE_BAR = 1
    PLOT_TYPE_HISTOGRAM = 2
    PLOT_TYPE_HEATMAP = 3

    def __init__(self, number_of_plots=1, plot_stretching=None, show_toolbar=False, parent=None):

        QWidget.__init__(self, parent)
        self.setWindowTitle('Plot')

        self.__show_toolbar = show_toolbar

        self.__plots = []
        self.__plot_infos = []

        self.__font = QFont( self.font().family(), 13, QFont.Bold )

        self.build_widget( number_of_plots, plot_stretching )

        self.on_draw()


    def show_toolbar(self):
        return self.__show_toolbar

    def get_plot_stretching(self):
        return self.__plot_stretching

    def get_number_of_plots(self):
        return self.__number_of_plots

    def set_number_of_plots(self, number_of_plots, plot_stretching=None):
        if  self.__number_of_plots != number_of_plots \
         or self.__plot_stretching != plot_stretching:
            self.rebuild_widget( number_of_plots, plot_stretching )


    def on_draw(self, plot_index=-1):

        if plot_index < 0:
            plot_indices = range( self.get_number_of_plots() )
        else:
            plot_indices = [ plot_index ]

        for plot_index in plot_indices:

            label, canvas, fig, axes, toolbar, widget = self.__plots[ plot_index ]

            plot_type, caption, data, mpl_kwargs = self.__plot_infos[ plot_index ]

            label.setText( caption )

            if plot_type == self.PLOT_TYPE_BAR:
                self.on_draw_barplot( fig, axes, data, mpl_kwargs )
            elif plot_type == self.PLOT_TYPE_HISTOGRAM:
                self.on_draw_histogram( fig, axes, data, mpl_kwargs )
            elif plot_type == self.PLOT_TYPE_HEATMAP:
                self.__clear_plot( plot_index )
                label, canvas, fig, axes, toolbar, widget = self.__plots[ plot_index ]
                self.on_draw_heatmap( fig, axes, data, mpl_kwargs )

            canvas.draw()


    def draw_histogram(self, plot_index, caption, values, bins, bin_labels=None, bin_rescale=None, **kwargs):
        data = ( values, bins, bin_labels, bin_rescale )
        self.__plot_infos[ plot_index ] = ( self.PLOT_TYPE_HISTOGRAM, caption, data, kwargs )
        self.on_draw( plot_index )

    def on_draw_histogram(self, fig, axes, data, mpl_kwargs):
        # Redraws the figure

        values, bins, bin_labels, bin_rescale = data

        axes.clear()

        if values != None:

            if bin_labels != None:
                x = numpy.arange( bins )
                tmp = numpy.zeros( ( bins, ), int )
                for v in x:
                    tmp[ v ] += numpy.sum( values[ : ] == v )

                if bin_rescale != None:
                    tmp = tmp * bin_rescale

                if not mpl_kwargs.has_key( 'facecolor' ): mpl_kwargs[ 'facecolor' ] = 'yellow'
                if not mpl_kwargs.has_key( 'alpha' ): mpl_kwargs[ 'alpha' ] = 0.75
                if not mpl_kwargs.has_key( 'align' ): mpl_kwargs[ 'align' ] = 'center'
                axes.bar( x, tmp, **mpl_kwargs )
                axes.set_xticks( x )
                axes.set_xticklabels( bin_labels )

            else:

                if not mpl_kwargs.has_key( 'facecolor' ): mpl_kwargs[ 'facecolor' ] = 'green'
                if not mpl_kwargs.has_key( 'alpha' ): mpl_kwargs[ 'alpha' ] = 0.75
                if not mpl_kwargs.has_key( 'align' ): mpl_kwargs[ 'align' ] = 'mid'
                axes.hist( values, bins, **mpl_kwargs )

            axes.grid( True )


    def draw_barplot(self, plot_index, caption, values, x=None, x_labels=None, **kwargs):
        data = ( values, x, x_labels )
        self.__plot_infos[ plot_index ] = ( self.PLOT_TYPE_BAR, caption, data, kwargs )
        self.on_draw( plot_index )

    def on_draw_barplot(self, fig, axes, data, mpl_kwargs):
        # Redraws the figure

        values, x, x_labels = data

        axes.clear()

        if values != None:

            if x != None and x_labels == None:
                x_labels = []
                for v in x:
                    x_labels.append( str( v ) )

            if x == None:
                x = numpy.arange( values.shape[0] )

            if not mpl_kwargs.has_key( 'facecolor' ): mpl_kwargs[ 'facecolor' ] = 'red'
            if not mpl_kwargs.has_key( 'alpha' ): mpl_kwargs[ 'alpha' ] = 0.75
            if not mpl_kwargs.has_key( 'align' ): mpl_kwargs[ 'align' ] = 'center'
            axes.bar( x, values, **mpl_kwargs )
            if x_labels != None:
                axes.set_xticks( x )
                axes.set_xticklabels( x_labels )

            axes.grid( True )


    def draw_heatmap(self, plot_index, caption, heatmap, x=None, x_labels=None, y=None, y_labels=None, **kwargs):
        data = ( heatmap, x, x_labels, y, y_labels )
        self.__plot_infos[ plot_index ] = ( self.PLOT_TYPE_HEATMAP, caption, data, kwargs )
        self.on_draw( plot_index )

    def on_draw_heatmap(self, fig, axes, data, mpl_kwargs):
        # Redraws the figure

        heatmap, x, x_labels, y, y_labels = data

        if heatmap != None:

            aximg = axes.imshow( heatmap, **mpl_kwargs )

            if x == None:
                x = numpy.arange( 0.5, heatmap.shape[1] - 1.5 )
                axes.set_xticks( x, minor=True )
                x = numpy.arange( heatmap.shape[1] )
                axes.set_xticks( x, minor=False )
                axes.set_xlim( -0.5, heatmap.shape[1] - 0.5)
            else:
                axes.set_xticks( x )

            if y == None:
                y = numpy.arange( 0.5, heatmap.shape[0] - 1.5 )
                axes.set_yticks( y, minor=True )
                y = numpy.arange( heatmap.shape[0] )
                axes.set_yticks( y, minor=False )
                axes.set_ylim( -0.5, heatmap.shape[0] - 0.5)
            else:
                axes.set_yticks( y )

            if x_labels != None:
                axes.set_xticklabels( x_labels )

            if y_labels != None:
                axes.set_yticklabels( y_labels )

            if not mpl_kwargs.has_key( 'color' ): mpl_kwargs[ 'color' ] = 'white'
            if not mpl_kwargs.has_key( 'linestyle' ): mpl_kwargs[ 'linestyle' ] = '-'
            if not mpl_kwargs.has_key( 'linewidth' ): mpl_kwargs[ 'linewidth' ] = 2
            if not mpl_kwargs.has_key( 'which' ): mpl_kwargs[ 'which' ] = 'minor'
            axes.grid( True )

            fig.colorbar( aximg )


    def __clear_plot(self, plot_index):

        plot = self.__plots[ plot_index ]
        label, canvas, fig, axes, toolbar, widget = plot

        del axes
        fig.clear()
        axes = fig.add_subplot(111)

        self.__plots[ plot_index ] = ( label, canvas, fig, axes, toolbar, widget )


    def build_widget(self, number_of_plots, plot_stretching):

        self.__widgets = []

        self.__top_vbox = QVBoxLayout()

        self.setLayout( self.__top_vbox )

        self.rebuild_widget( number_of_plots, plot_stretching )

    def rebuild_widget(self, number_of_plots, plot_stretching):

        self.__number_of_plots = number_of_plots
        self.__plot_stretching = plot_stretching

        ids = []
        if self.__top_vbox.count() < self.__number_of_plots:
            ids = range( self.__top_vbox.count(), self.__number_of_plots )

        for i in ids:

            label = QLabel()
            label.setFont( self.__font )
            label.setAlignment( Qt.AlignCenter )

            # Create the mpl Figure and FigCanvas objects. 
            # 5x4 inches, 100 dots-per-inch
            #
            dpi = 100
            #self.fig = Figure((5.0, 4.0), dpi=self.dpi)
            fig = pyplot.figure()
            canvas = FigureCanvas( fig )
            canvas.setParent( self )
    
            # Since we have only one plot, we can use add_axes 
            # instead of add_subplot, but then the subplot
            # configuration tool in the navigation toolbar wouldn't
            # work.
            #
            axes = fig.add_subplot(111)
    
            # Create the navigation toolbar, tied to the canvas
            #
            mpl_toolbar = NavigationToolbar(canvas, self, False)
    
            if self.__show_toolbar:
                mpl_toolbar.show()
            else:
                mpl_toolbar.hide()

            # Other GUI controls
            #
    
            tmp_vbox = QVBoxLayout()
            tmp_vbox.addWidget( label )
            tmp_vbox.addWidget( canvas, 1 )
            tmp_vbox.addWidget( mpl_toolbar )

            widget = QWidget()
            widget.setLayout( tmp_vbox )

            self.__plots.append( ( label, canvas, fig, axes, mpl_toolbar, widget ) )

            self.__plot_infos.append( ( self.PLOT_TYPE_NONE, '', None, {} ) )

            self.__top_vbox.addWidget( widget )

        for i in xrange( self.__number_of_plots ):

            stretch = 0
            if plot_stretching != None:
                stretch = plot_stretching[ i ]
                self.__top_vbox.setStretch( i, stretch )

            plot = self.__plots[ i ]
            label, canvas, fig, axes, mpl_toolbar, widget = plot
            widget.show()

        for i in xrange( self.__number_of_plots, self.__top_vbox.count() ):

            plot = self.__plots[ i ]
            label, canvas, fig, axes, mpl_toolbar, widget = plot
            widget.hide()
