# -*- coding: utf-8 -*-

"""
cluster_profiles_window.py -- Visualization of cluster profiles.
"""

# This software is distributed under the FreeBSD License.
# See the accompanying file LICENSE for details.
# 
# Copyright 2011 Benjamin Hepp

import sys, os, random
from PyQt4.QtCore import *
from PyQt4.QtGui import *

import numpy
import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import pyplot

from plot_window import PlotWindow


class ClusterProfilesWindow(PlotWindow):

    PROFILE_COMPARISON_PLOT_ID = 0
    DISTANCE_HEATMAP_PLOT_ID = 1


    def __init__(self, labels, clusterProfiles, distanceMatrix, plot_stretching=None, show_toolbar=False, parent=None):

        PlotWindow.__init__(self, 2, plot_stretching, show_toolbar, parent)
        self.setWindowTitle('Cluster profiles')

        mask1 = numpy.isfinite(distanceMatrix)[0]
        mask2 = numpy.isfinite(distanceMatrix)[:,0]
        valid_mask = numpy.logical_and(mask1, mask2)
        valid_indices = numpy.arange(valid_mask.shape[0])[valid_mask]
        self.__labels = []
        for i in valid_indices:
            self.__labels.append(labels[i])
        self.__clusterProfiles = clusterProfiles[valid_mask]
        self.__distanceMatrix = distanceMatrix[valid_mask][:,valid_mask]

        self.__create_distance_heatmap_plot()


    def __draw_profile_comparison(self, fig, axes, custom_data, **kwargs):

        if 'bottom_shift' in kwargs:
            bottom_shift = kwargs['bottom_shift']
            del kwargs['bottom_shift']
        else:
            bottom_shift = None

        min_profile, profile1, profile2, x, x_labels = custom_data

        axes.clear()

        if 'facecolor' in kwargs:
            del kwargs['facecolor']
        if not kwargs.has_key('alpha'): kwargs['alpha'] = 1.0
        if not kwargs.has_key('align'): kwargs['align'] = 'center'

        axes.bar(x, min_profile, color='yellow', **kwargs)

        axes.bar(x, profile1 - min_profile, bottom=min_profile, color='green', **kwargs)

        axes.bar(x, profile2 - min_profile, bottom=min_profile, color='red', **kwargs)

        axes.set_xlim(-1, x.shape[0])

        if x_labels != None:
            axes.set_xticks(x)
            x_labels = axes.set_xticklabels(x_labels, rotation='270')
            if bottom_shift != None and bottom_shift > fig.subplotpars.bottom:
                fig.subplots_adjust(bottom=bottom_shift)

            #axes.grid(True)

        #return x_labels

    def __create_profile_comparison_plot(self, treatmentId1, treatmentId2):

        profile1 = self.__clusterProfiles[treatmentId1]
        profile2 = self.__clusterProfiles[treatmentId2]

        profile1 = profile1 / numpy.sum(profile1)
        profile2 = profile2 / numpy.sum(profile2)

        min_profile = numpy.min([profile1, profile2], axis=0)

        x = numpy.arange(self.__clusterProfiles.shape[1])

        x_labels = []
        step = int(profile1.shape[0] / 40 ) + 1
        for j in xrange(x.shape[0]):
            if j % step == 0:
                x_labels.append(str(x[j]))
            else:
                x_labels.append('')

        caption = "Profile comparison of treatment '%s' and '%s'" % (self.__labels[treatmentId1], self.__labels[treatmentId2])

        custom_data = [min_profile, profile1, profile2, x, x_labels]

        self.draw_custom(self.PROFILE_COMPARISON_PLOT_ID, caption, self.__draw_profile_comparison, [custom_data])

        #self.on_draw(self.PROFILE_COMPARISON_PLOT_ID)


    def __draw_distance_heatmap(self, fig, axes, **kwargs):

        if 'bottom_shift' in kwargs:
            bottom_shift = kwargs['bottom_shift']
            del kwargs['bottom_shift']
        else:
            bottom_shift = None

        aximg = axes.imshow(self.__distanceMatrix, interpolation='nearest')
    
        axes.set_title('Distance heatmap')

        x = numpy.arange(self.__distanceMatrix.shape[1])
        axes.set_xticks(x, minor=False)
        axes.set_xlim(-0.5, self.__distanceMatrix.shape[1] - 0.5)
    
        #y = numpy.arange(0.5, profileHeatmap.shape[0] - 1.5)
        #axes.set_yticks(y, minor=True)
        y = numpy.arange(self.__distanceMatrix.shape[0])
        axes.set_yticks(y, minor=False)
        axes.set_ylim(-0.5, self.__distanceMatrix.shape[0] - 0.5)
    
        labels = []
        for i in xrange(self.__distanceMatrix.shape[0]):
            labels.append(self.__labels[i])
    
        x_labels = axes.set_xticklabels(labels, rotation='270')
        axes.set_yticklabels(labels, rotation='0')
    
        if bottom_shift != None and bottom_shift > fig.subplotpars.bottom:
            fig.subplots_adjust(bottom=bottom_shift)

        #if not mpl_kwargs.has_key('color'): mpl_kwargs['color'] = 'white'
        #if not mpl_kwargs.has_key('linestyle'): mpl_kwargs['linestyle'] = '-'
        #if not mpl_kwargs.has_key('linewidth'): mpl_kwargs['linewidth'] = 2
        #if not mpl_kwargs.has_key('which'): mpl_kwargs['which'] = 'minor'
        #axes.grid(True)
    
        fig.colorbar(aximg)
    
        return x_labels


    def __on_heatmap_button_press(self, event):

        self.__heatmap_button_pressed_pos = (event.x, event.y)

    def __on_heatmap_button_release(self, event):

        if self.__heatmap_button_pressed_pos == None:
            return
        if  self.__heatmap_button_pressed_pos != (event.x, event.y):
            self.__heatmap_button_pressed_pos = None
            return

        self.__heatmap_button_pressed_pos = None

        self.__on_heatmap_button_clicked(event)

    def __on_heatmap_button_clicked(self, event):

        mx = event.xdata
        my = event.ydata

        if mx != None and my != None:

            x = int(mx + 0.5)
            y = int(my + 0.5)

            self.__heatmap_selected_x = x
            self.__heatmap_selected_y = y

            #self.__create_profile_comparison_plot(x, y)

    def __on_heatmap_motion_notify(self, event):

        mx = event.xdata
        my = event.ydata

        #print 'mx:', mx, 'my:', my

        if mx != None and my != None:

            x = int(mx + 0.5)
            y = int(my + 0.5)

            #print 'x:', x, 'y:', y

            if (x != self.__heatmap_current_x) or (y != self.__heatmap_current_y):
                self.__create_profile_comparison_plot(x, y)

            self.__heatmap_current_x = x
            self.__heatmap_current_y = y

        else:

            self.__on_heatmap_leave(event)

    def __on_heatmap_leave(self, event):

        if (self.__heatmap_current_x != self.__heatmap_selected_x) \
        or (self.__heatmap_current_y != self.__heatmap_selected_y):
            self.__create_profile_comparison_plot(
                self.__heatmap_selected_x,
                self.__heatmap_selected_y
           )

        self.__heatmap_current_x = self.__heatmap_selected_x
        self.__heatmap_current_y = self.__heatmap_selected_y


    def __create_distance_heatmap_plot(self):

        caption = 'Heatmap of profile distances'

        self.draw_custom(self.DISTANCE_HEATMAP_PLOT_ID, caption, self.__draw_distance_heatmap)


        self.__heatmap_selected_x = 0
        self.__heatmap_selected_y = 1

        self.__heatmap_current_x = self.__heatmap_selected_x
        self.__heatmap_current_y = self.__heatmap_selected_y

        self.__create_profile_comparison_plot(self.__heatmap_selected_x, self.__heatmap_selected_y)


        self.__heatmap_button_pressed_pos = None

        self.connect_event_handler(self.DISTANCE_HEATMAP_PLOT_ID, 'button_press_event', self.__on_heatmap_button_press)
        self.connect_event_handler(self.DISTANCE_HEATMAP_PLOT_ID, 'button_release_event', self.__on_heatmap_button_release)

        # Bind the 'motion_notify' event for moving the mouse
        self.connect_event_handler(self.DISTANCE_HEATMAP_PLOT_ID, 'motion_notify_event', self.__on_heatmap_motion_notify)

        # Bind the 'axes_leave' event for leaving a figure with the mouse
        self.connect_event_handler(self.DISTANCE_HEATMAP_PLOT_ID, 'axes_leave_event', self.__on_heatmap_leave)

        # Bind the 'figure_leave' event for leaving a figure with the mouse
        self.connect_event_handler(self.DISTANCE_HEATMAP_PLOT_ID, 'figure_leave_event', self.__on_heatmap_leave)


