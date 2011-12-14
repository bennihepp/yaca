# -*- coding: utf-8 -*-

"""
treatment_comparison_window.py -- Visualization of treatment comparisons.
"""

# This software is distributed under the FreeBSD License.
# See the accompanying file LICENSE for details.
# 
# Copyright 2011 Benjamin Hepp

import sys, os, random
from PyQt4.QtCore import *
from PyQt4.QtGui import *

import itertools
import numpy
import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import pyplot

from plot_window import PlotWindow, MultiPagePlotWindow

class TreatmentComparisonWindow(MultiPagePlotWindow):

    def __init__(self, pageList, plotLabels, labels,
                 distanceMatrix, neighbourList=None,
                 plot_stretching=None, show_toolbar=None,
                 show_menu=False, parent=None):

        pageToIds = []
        i = 0
        for pageLen,pageLabel in pageList:
            if len(pageToIds) > 0:
                i = pageToIds[-1][-1] + 1
            else:
                i = 0
            pageToIds.append(range(i, i+pageLen))

        #print pageToIds
        #print pageList
        #print plotLabels

        show_toolbar = len(pageList) * [True]
        MultiPagePlotWindow.__init__(self, pageList, show_toolbar=show_toolbar,
                                     show_menu=show_menu, parent=parent)
        self.setWindowTitle('Treatment comparison')

        self.__distanceMatrix = distanceMatrix
        self.__maxDistance = numpy.max(self.__distanceMatrix)

        self.__create_treatment_comparison_plot(pageToIds, neighbourList, plotLabels, labels)


    def __draw_treatment_comparison(self, fig, axes, trId, trNeighbourIds, distances, labels, **kwargs):

        #trId, trNeighbourIds, distances, labels = custom_data
        maxDistance = self.__maxDistance

        axes.clear()

        if 'facecolor' in kwargs:
            del kwargs[ 'facecolor' ]
        if not kwargs.has_key( 'alpha' ): kwargs[ 'alpha' ] = 1.0
        if not kwargs.has_key( 'align' ): kwargs[ 'align' ] = 'center'

        axes.set_title('Distance from %s' % labels[trId])

        axes.set_xlim(-0.10 * maxDistance, 1.10 * maxDistance)
        axes.set_ylim(-0.5, 0.5)

        yticks = axes.yaxis.get_major_ticks() + axes.yaxis.get_minor_ticks()
        for tick in yticks:
            tick.tick1On = False
            tick.tick2On = False
            tick.label1On = False
            tick.label2On = False

        xticks = axes.xaxis.get_major_ticks() + axes.xaxis.get_minor_ticks()
        for tick in xticks:
            tick.tick1On = False
            tick.tick2On = False
            tick.label1On = False
            tick.label2On = False

        main_line = matplotlib.lines.Line2D(
            [-0.05 * maxDistance, 1.05 * maxDistance],
            [0,0],
            color='black'
        )
        axes.add_artist(main_line)
        #arr = matplotlib.patches.Arrow(
        #    1.0 * maxDistance, 0,
        #    0.05 * maxDistance, 0,
        #    linewidth=0.03,
        #    width=0.03,
        #    color='black'
        #)
        #axes.add_artist(arr)
        for tick in axes.get_xticks():
            if tick >= 0 and tick <= main_line.get_xdata()[1]:
                l = matplotlib.lines.Line2D(
                    [tick, tick],
                    [-0.1, 0],
                    color='black',
                )
                axes.add_artist(l)
                t = matplotlib.text.Text(
                    tick, -0.15,
                    '%.2f' % tick,
                    color='black',
                    verticalalignment='center',
                    horizontalalignment='center',
                    transform = axes.transData
                )
                axes.add_artist(t)

        for i,distance in enumerate(distances):
            if i == trId:
                color = 'green'
            elif i in trNeighbourIds:
                color = 'blue'
            else:
                color = 'red'
            l = matplotlib.lines.Line2D(
                [distance, distance],
                [0, 0.15],
                linewidth=2.0,
                color=color
            )
            axes.add_artist(l)
            t = matplotlib.text.Text(
                distance, +0.2,
                labels[i],
                color='black',
                verticalalignment='left',
                horizontalalignment='center',
                transform = axes.transData,
                rotation=90
            )
            axes.add_artist(t)

        selfDistance = distances[trId]
        neighbourDistance = numpy.mean(
            [distances[i] for i in itertools.ifilter(
                lambda j: j != trId and j in trNeighbourIds, xrange(len(distances)))
            ]
        )
        otherDistance = numpy.mean(
            [distances[i] for i in itertools.ifilter(
                lambda j: j != trId and j not in trNeighbourIds, xrange(len(distances)))
            ]
        )
        l1 = matplotlib.lines.Line2D(
            [selfDistance, selfDistance],
            [-0.27,-0.31],
            color='blue'
        )
        l2 = matplotlib.lines.Line2D(
            [neighbourDistance, neighbourDistance],
            [-0.27,-0.31],
            color='blue'
        )
        l3 = matplotlib.lines.Line2D(
            [selfDistance, neighbourDistance],
            [-0.29,-0.29],
            color='blue'
        )
        for l in [l1,l2,l3]: axes.add_artist(l)

        l1 = matplotlib.lines.Line2D(
            [selfDistance, selfDistance],
            [-0.20,-0.24],
            color='green'
        )
        l2 = matplotlib.lines.Line2D(
            [0, 0],
            [-0.20,-0.24],
            color='green'
        )
        l3 = matplotlib.lines.Line2D(
            [selfDistance, 0],
            [-0.22,-0.22],
            color='green'
        )
        for l in [l1,l2,l3]: axes.add_artist(l)

        l1 = matplotlib.lines.Line2D(
            [selfDistance, selfDistance],
            [-0.34,-0.38],
            color='red'
        )
        l2 = matplotlib.lines.Line2D(
            [otherDistance, otherDistance],
            [-0.34,-0.38],
            color='red'
        )
        l3 = matplotlib.lines.Line2D(
            [selfDistance, otherDistance],
            [-0.36,-0.36],
            color='red'
        )
        for l in [l1,l2,l3]: axes.add_artist(l)

        #if x_labels != None:
        #    axes.set_xticks( x )
        #    x_labels = axes.set_xticklabels( x_labels, rotation='270' )
        #    if bottom_shift != None and fig.subplotpars.bottom < bottom_shift:
        #        fig.subplots_adjust( bottom=bottom_shift )

            #axes.grid( True )

        #return x_labels

    def __create_treatment_comparison_plot(self, pageToIds, neighbourList, plotLabels, labels):

        #tr_labels = []
        #for tr in pdc.treatments:
        #    for repl in pdc.replicates:
        #        tr_labels.append(tr.name)

        #tr_labels = []
        #for i in xrange(len(pageToIds)):
        #    tr_labels.append(self.__pdc.treatments[i].name)

        for page,ids in enumerate(pageToIds):
            #self.page = page
            for i,id in enumerate(ids):
                if neighbourList != None:
                    neighbourIds = neighbourList[id]
                else:
                    neighbourIds = ids
                self.get_plot_window(page).draw_custom(
                    i, plotLabels[id], self.__draw_treatment_comparison,
                    [id, neighbourIds, self.__distanceMatrix[id], labels]
                )

        #self.page = page
