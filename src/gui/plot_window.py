# -*- coding: utf-8 -*-

"""
plot_window.py -- Windows for showing various plots.
"""

# This software is distributed under the FreeBSD License.
# See the accompanying file LICENSE for details.
# 
# Copyright 2011 Benjamin Hepp

import sys, os, random
from PyQt4.QtCore import *
from PyQt4.QtGui import *

import functools
import numpy
import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import pyplot

import print_engine
from multipage_window import MultiPageWindow

class MultiPagePlotWindow(MultiPageWindow):

    def __init__(self, pageList=[(1,'Page 1')], plot_stretching=None, show_toolbar=None, show_menu=True, parent=None, title='Plot'):
        pageNum, pageTitles = zip(*pageList)
        MultiPageWindow.__init__(self, pageTitles, title=title)
        if plot_stretching == None:
            plot_stretching = len(pageList) * [None]
        if show_toolbar == None:
            show_toolbar = len(pageList) * [False]
        elif type(show_toolbar) == bool:
            show_toolbar = len(pageList) * [show_toolbar]
        self.__build_widget(pageList, plot_stretching, show_toolbar, show_menu)

    plot_window = MultiPageWindow.child_window
    get_plot_window = MultiPageWindow.get_child_window

    def on_save_page(self, page):
        plot_window = self.get_plot_window(page)
        return plot_window.on_save_all_plots()

    def on_save_all_pages(self):
        file_choices = "PDF file (*.pdf)"
        path = unicode(QFileDialog.getSaveFileName(self, 
                        'Save pages as PDF', '', 
                        file_choices))
        if path:
            plots_per_page, ok = \
                          QInputDialog.getInt(self, 'Save all plots', 'Number of plots per page:',
                                              value=2, min=0,
                                              max=10)
            if ok:
                #self.parent().statusBar().showMessage('Saving all pages to %s' % path)
                pdf = print_engine.PdfDocument(path)
                for page in xrange(self.page_count):
                    plot_window = self.get_plot_window(page)
                    plot_window.on_save_all_plots(pdf, plots_per_page)
                pdf.close()
                #self.parent().statusBar().showMessage('Saved all pages to %s' % path, 2000)
                return True
        return False

    def __build_widget(self, pageList, plot_stretching, show_toolbar, show_menu):
        for i, (numOfPlots, pageName) in enumerate(pageList):
            plotWindow = PlotWindow(numOfPlots, plot_stretching[i], show_toolbar[i])
            self.set_child_window(i, plotWindow)

        if show_menu:
            menubar = QMenuBar()
            save_menu = menubar.addMenu("&Save")
            menu_items = []
            for page, (numOfPlots, pageName) in enumerate(pageList):
                menu_name = "Save &page '%s' [%d]" % (pageName, page+1)
                if len(pageName) == 0:
                    menu_name = "Save &page #%d" % (page+1)
                save_page_action = self.__make_action(menu_name,
                                                     shortcut="Ctrl+P",
                                                     slot=functools.partial(self.on_save_page, page))
                menu_items.append(save_page_action)
            menu_items.append(None)
            save_all_action = self.__make_action("Save &all pages",
                shortcut="Ctrl+A", slot=self.on_save_all_pages)
            menu_items.append(save_all_action)
            self.__add_actions(save_menu, menu_items)
            self.layout().setMenuBar(menubar)

    def __add_actions(self, target, actions):
        for action in actions:
            if action is None:
                target.addSeparator()
            else:
                target.addAction(action)

    def __make_action( self, text, slot=None, shortcut=None, 
                        icon=None, tip=None, checkable=False, 
                        signal="triggered()"):
        action = QAction(text, self)
        if icon is not None:
            action.setIcon(QIcon(":/%s.png" % icon))
        if shortcut is not None:
            action.setShortcut(shortcut)
        if tip is not None:
            action.setToolTip(tip)
            action.setStatusTip(tip)
        if slot is not None:
            self.connect(action, SIGNAL(signal), slot)
        if checkable:
            action.setCheckable(True)
        return action


class PlotWindow(QWidget):

    PLOT_TYPE_NONE = 0
    PLOT_TYPE_BAR = 1
    PLOT_TYPE_HISTOGRAM = 2
    PLOT_TYPE_HEATMAP = 3
    PLOT_TYPE_LINE = 4
    PLOT_TYPE_CONTOUR = 5
    PLOT_TYPE_CUSTOM = 100

    def __init__(self, number_of_plots=1, plot_stretching=None, show_toolbar=False, parent=None, title='Plot', show_menu=True):

        QWidget.__init__(self, parent)
        self.setWindowTitle(title)

        self.__show_toolbar = show_toolbar
        self.__show_menu = show_menu

        self.__plots = []
        self.__plot_infos = []

        self.__font = QFont(self.font().family(), 13, QFont.Bold)

        self.build_widget(number_of_plots, plot_stretching)

        self.on_draw()

    def connect_event_handler(self, plot_index, event, handler):

        label, canvas, fig, axes, toolbar, widget = self.__plots[plot_index]

        canvas.mpl_connect(event, handler)


    def show_toolbar(self):
        return self.__show_toolbar

    def get_plot_stretching(self):
        return self.__plot_stretching

    def get_number_of_plots(self):
        return self.__number_of_plots

    def set_number_of_plots(self, number_of_plots, plot_stretching=None):
        if  self.__number_of_plots != number_of_plots \
         or self.__plot_stretching != plot_stretching:
            self.rebuild_widget(number_of_plots, plot_stretching)


    def set_caption(self, plot_index, caption):
        plot_type, old_caption, data, mpl_kwargs = self.__plot_infos[plot_index]
        self.__plot_infos[plot_index] = (plot_type, caption, data, mpl_kwargs)
        self.__plots[plot_index][0].setText(caption)
    def get_figure(self, plot_index):
        label, canvas, fig, axes, toolbar, widget = self.__plots[plot_index]
        return fig
    def get_axes(self, plot_index):
        label, canvas, fig, axes, toolbar, widget = self.__plots[plot_index]
        return axes
    def get_figure_and_axes(self, plot_index):
        label, canvas, fig, axes, toolbar, widget = self.__plots[plot_index]
        return fig, axes

    def __compute_labels_bbox(self, fig, labels):
        import matplotlib.transforms as mtransforms
        bboxes = []
        for label in labels:
            bbox = label.get_window_extent()
            bboxi = bbox.inverse_transformed(fig.transFigure)
            bboxes.append(bboxi)
        bbox = mtransforms.Bbox.union(bboxes)
        return bbox


    def on_draw(self, plot_index=-1, overwrite_fig=None, overwrite_axes=None, draw_canvas=True, print_mode=False):

        if plot_index < 0:
            plot_indices = range(self.get_number_of_plots())
        else:
            plot_indices = [plot_index]

        for plot_index in plot_indices:

            label, canvas, fig, axes, toolbar, widget = self.__plots[plot_index]

            if overwrite_fig != None:
                fig = overwrite_fig
            if overwrite_axes != None:
                axes = overwrite_axes

            plot_type, caption, data, mpl_kwargs = self.__plot_infos[plot_index]

            label.setText(caption)

            labels = None
            bottom_shift = 0.0
            go_on = True

            while go_on:

                go_on = False

                old_labels = labels

                if plot_type == self.PLOT_TYPE_BAR:
                    labels = self.on_draw_barplot(fig, axes, data, bottom_shift, **mpl_kwargs)
                elif plot_type == self.PLOT_TYPE_HISTOGRAM:
                    labels = self.on_draw_histogram(fig, axes, data, bottom_shift, **mpl_kwargs)
                elif plot_type == self.PLOT_TYPE_HEATMAP:
                    self.__clear_plot(plot_index)
                    label, canvas, fig, axes, toolbar, widget = self.__plots[plot_index]
                    if overwrite_fig != None:
                        fig = overwrite_fig
                    if overwrite_axes != None:
                        axes = overwrite_axes
                    labels = self.on_draw_heatmap(fig, axes, data, bottom_shift, **mpl_kwargs)
                elif plot_type == self.PLOT_TYPE_LINE:
                    labels = self.on_draw_lineplot(fig, axes, data, bottom_shift, **mpl_kwargs)
                elif plot_type == self.PLOT_TYPE_CONTOUR:
                    labels = self.on_draw_contour(fig, axes, data, bottom_shift, **mpl_kwargs)
                elif plot_type == self.PLOT_TYPE_CUSTOM:
                    self.__clear_plot(plot_index)
                    label, canvas, fig, axes, toolbar, widget = self.__plots[plot_index]
                    if overwrite_fig != None:
                        fig = overwrite_fig
                    if overwrite_axes != None:
                        axes = overwrite_axes
                    drawing_method, custom_args, custom_kwargs, want_figure, want_bottomshift = data
                    if want_bottomshift and bottom_shift > 0.0:
                        custom_kwargs['bottom_shift'] = bottom_shift
                    args = [fig, axes] + custom_args
                    if not want_figure:
                        del args[0]
                    labels = drawing_method(*args, **custom_kwargs)

                if draw_canvas:
                    canvas.draw()

                if not print_mode:
                    if labels != None:
                        bbox = self.__compute_labels_bbox(fig, labels)
                        bottom_shift = 1.1 * bbox.height

                    if old_labels == None and labels != None:
                        go_on = True


    def draw_custom(self, plot_index, caption, drawing_method,
                    custom_args=[], custom_kwargs={}, want_figure=True, want_bottomshift=False, **kwargs):
        data = (drawing_method, custom_args, custom_kwargs, want_figure, want_bottomshift)
        self.__plot_infos[plot_index] = (self.PLOT_TYPE_CUSTOM, caption, data, kwargs)
        self.on_draw(plot_index)

    def draw_histogram(self, plot_index, caption, values, bins, bin_labels=None, bin_rescale=None, **kwargs):
        data = (values, bins, bin_labels, bin_rescale)
        self.__plot_infos[plot_index] = (self.PLOT_TYPE_HISTOGRAM, caption, data, kwargs)
        self.on_draw(plot_index)

    def on_draw_histogram(self, fig, axes, data, bottom_shift=None, **mpl_kwargs):
        # Redraws the figure

        values, bins, bin_labels, bin_rescale = data

        axes.clear()

        if values != None:

            if bin_labels != None:
                x = numpy.arange(bins)
                tmp = numpy.zeros((bins,), int)
                for v in x:
                    tmp[v] += numpy.sum(values[:] == v)

                if bin_rescale != None:
                    tmp = tmp * bin_rescale

                if not mpl_kwargs.has_key('facecolor'): mpl_kwargs['facecolor'] = 'yellow'
                if not mpl_kwargs.has_key('alpha'): mpl_kwargs['alpha'] = 0.75
                if not mpl_kwargs.has_key('align'): mpl_kwargs['align'] = 'center'
                axes.bar(x, tmp, **mpl_kwargs)
                axes.set_xticks(x)
                bin_labels = axes.set_xticklabels(bin_labels, rotation=270)
                if bottom_shift != None and fig.subplotpars.bottom < bottom_shift:
                    fig.subplots_adjust(bottom=bottom_shift)

            else:

                if not mpl_kwargs.has_key('facecolor'): mpl_kwargs['facecolor'] = 'green'
                if not mpl_kwargs.has_key('alpha'): mpl_kwargs['alpha'] = 0.75
                if not mpl_kwargs.has_key('align'): mpl_kwargs['align'] = 'mid'
                axes.hist(values, bins, **mpl_kwargs)

            axes.grid(True)

        return bin_labels


    def draw_barplot(self, plot_index, caption, values, x=None, x_labels=None, **kwargs):
        data = (values, x, x_labels)
        self.__plot_infos[plot_index] = (self.PLOT_TYPE_BAR, caption, data, kwargs)
        self.on_draw(plot_index)

    def on_draw_barplot(self, fig, axes, data, bottom_shift=None, **mpl_kwargs):
        # Redraws the figure

        values, x, x_labels = data

        axes.clear()

        if values != None:

            if x != None and x_labels == None:
                x_labels = []
                for v in x:
                    x_labels.append(str(v))

            if x == None:
                x = numpy.arange(values.shape[0])

            if not mpl_kwargs.has_key('facecolor'): mpl_kwargs['facecolor'] = 'red'
            if not mpl_kwargs.has_key('alpha'): mpl_kwargs['alpha'] = 0.75
            if not mpl_kwargs.has_key('align'): mpl_kwargs['align'] = 'center'
            axes.bar(x, values, **mpl_kwargs)
            if x_labels != None:
                axes.set_xticks(x)
                x_labels = axes.set_xticklabels(x_labels, rotation='270')
                if bottom_shift != None and fig.subplotpars.bottom < bottom_shift:
                    fig.subplots_adjust(bottom=bottom_shift)

            axes.grid(True)

        return x_labels


    def draw_heatmap(self, plot_index, caption, heatmap, x=None, x_labels=None, y=None, y_labels=None, **kwargs):
        data = (heatmap, x, x_labels, y, y_labels)
        self.__plot_infos[plot_index] = (self.PLOT_TYPE_HEATMAP, caption, data, kwargs)
        self.on_draw(plot_index)

    def on_draw_heatmap(self, fig, axes, data, bottom_shift=None, **mpl_kwargs):
        # Redraws the figure

        heatmap, x, x_labels, y, y_labels = data

        if heatmap != None:

            aximg = axes.imshow(heatmap, **mpl_kwargs)

            if x == None:
                x = numpy.arange(0.5, heatmap.shape[1] - 1.5)
                axes.set_xticks(x, minor=True)
                x = numpy.arange(heatmap.shape[1])
                axes.set_xticks(x, minor=False)
                axes.set_xlim(-0.5, heatmap.shape[1] - 0.5)
            else:
                axes.set_xticks(x)

            if y == None:
                y = numpy.arange(0.5, heatmap.shape[0] - 1.5)
                axes.set_yticks(y, minor=True)
                y = numpy.arange(heatmap.shape[0])
                axes.set_yticks(y, minor=False)
                axes.set_ylim(-0.5, heatmap.shape[0] - 0.5)
            else:
                axes.set_yticks(y)

            if x_labels == None:
                x_labels = []
                for i in x:
                    x_labels.append(str(i))

            x_labels = axes.set_xticklabels(x_labels, rotation='270')
            if bottom_shift != None and fig.subplotpars.bottom < bottom_shift:
                fig.subplots_adjust(bottom=bottom_shift)

            if y_labels == None:
                y_labels = []
                for i in y:
                    y_labels.append(str(i))

            y_labels = axes.set_yticklabels(y_labels, rotation='0')

            if not mpl_kwargs.has_key('color'): mpl_kwargs['color'] = 'white'
            if not mpl_kwargs.has_key('linestyle'): mpl_kwargs['linestyle'] = '-'
            if not mpl_kwargs.has_key('linewidth'): mpl_kwargs['linewidth'] = 2
            if not mpl_kwargs.has_key('which'): mpl_kwargs['which'] = 'minor'
            axes.grid(True)

            fig.colorbar(aximg)

        return x_labels


    def draw_lineplot(self, plot_index, caption, x, y, marking='-', **kwargs):
        data = (x, y, marking)
        self.__plot_infos[plot_index] = (self.PLOT_TYPE_LINE, caption, data, kwargs)
        self.on_draw(plot_index)

    def on_draw_lineplot(self, fig, axes, data, bottom_shift=None, **mpl_kwargs):
        # Redraws the figure

        x, y, marking = data

        axes.clear()

        if x == None:
            x = numpy.arange(y.shape[0])

        if not mpl_kwargs.has_key('color'): mpl_kwargs['color'] = 'red'
        if not mpl_kwargs.has_key('alpha'): mpl_kwargs['alpha'] = 0.75
        if not mpl_kwargs.has_key('antialiased'): mpl_kwargs['antialiased'] = True
        #if not mpl_kwargs.has_key('marker'): mpl_kwargs['marker'] = 'None'
        #if not mpl_kwargs.has_key('linestyle'): mpl_kwargs['linestyle'] = '-'

        axes.plot(x, y, marking, **mpl_kwargs)

        x = axes.get_xticks()
        x_labels = []
        for i in x:
            x_labels.append(str(i))
        x_labels = axes.set_xticklabels(x_labels, rotation='270')
        if bottom_shift != None and fig.subplotpars.bottom < bottom_shift:
            fig.subplots_adjust(bottom=bottom_shift)

        axes.grid(True)

        return x_labels


    def draw_contour(self, plot_index, caption, XYZ, labels, colors, marks=[],
                     title=None, xlabel=None, ylabel=None, **kwargs):
        data = (XYZ, labels, colors, marks, title, xlabel, ylabel)
        self.__plot_infos[plot_index] = (self.PLOT_TYPE_CONTOUR, caption, data, kwargs)
        self.on_draw(plot_index)

    def on_draw_contour(self, fig, axes, data, bottom_shift=None, **mpl_kwargs):
        # Redraws the figure

        XYZ, labels, colors, marks, title, xlabel, ylabel = data

        axes.clear()
        axes.set_xlim(-5,5)
        axes.set_ylim(-5,5)
        if title != None:
            axes.set_title(title)
        if xlabel != None:
            axes.set_xlabel(xlabel)
        if ylabel != None:
            axes.set_ylabel(ylabel)
        lines = []
        for (X,Y,Z),color in zip(XYZ,colors):
            CS = axes.contour(X, Y, Z, colors=color, antialiased=True, **mpl_kwargs)
            #axes.clabel(CS, inline=True)
            lines.append(axes.plot([0],[0],color=color))
        axes.legend(lines, labels)
        if len(marks) > 0:
            axes.scatter(marks[:,0], marks[:,1], marker='x', color='black')

    def __clear_plot(self, plot_index):

        plot = self.__plots[plot_index]
        label, canvas, fig, axes, toolbar, widget = plot

        del axes
        fig.clear()
        axes = fig.add_subplot(111)

        self.__plots[plot_index] = (label, canvas, fig, axes, toolbar, widget)

    def on_save_plot(self, plot_index, pdfDocument=None):
        if pdfDocument == None:
            file_choices = "PDF file (*.pdf)"
            path = unicode(QFileDialog.getSaveFileName(self, 
                            'Save plot as PDF', '', 
                            file_choices))
            if path:
                pdfDocument = print_engine.PdfDocument(path)
        if pdfDocument != None:
            #self.parent().statusBar().showMessage('Saving plot to %s' % path)
            fig = pdfDocument.next_page()
            axes = pdfDocument.begin_next_plot()
            self.on_draw(plot_index, fig, axes, draw_canvas=False, print_mode=True)
            pdfDocument.end()
            pdfDocument.close()
            #self.parent().statusBar().showMessage('Saved plot to %s' % path, 2000)
            return True
        else:
            return False

    def on_save_all_plots(self, pdfDocument=None, plots_per_page=None):
        closePdfDocument = False
        if pdfDocument == None:
            file_choices = "PDF file (*.pdf)"
            path = unicode(QFileDialog.getSaveFileName(self, 
                            'Save plots as PDF', '', 
                            file_choices))
            if path:
                pdfDocument = print_engine.PdfDocument(path)
                closePdfDocument = True
        if pdfDocument != None:
            #self.parent().statusBar().showMessage('Saving all plots to %s' % path)
            ok = True
            if plots_per_page == None:
                plots_per_page, ok = \
                              QInputDialog.getInt(self, 'Save all plots', 'Number of plots per page:',
                                                  value=self.get_number_of_plots(), min=0,
                                                  max=self.get_number_of_plots())
            if ok:
                pdfDocument.begin()
                fig = pdfDocument.next_page(rows=plots_per_page, cols=1)
                for plot_index in xrange(self.get_number_of_plots()):
                    axes = pdfDocument.next_plot()
                    self.on_draw(plot_index, fig, axes, draw_canvas=False, print_mode=True)
                pdfDocument.end()
                if closePdfDocument:
                    pdfDocument.close()
                #self.parent().statusBar().showMessage('Saved all plots to %s' % path, 2000)
                return True
        return False

    def build_widget(self, number_of_plots, plot_stretching):

        self.__widgets = []

        self.__top_vbox = QVBoxLayout()

        self.setLayout(self.__top_vbox)

        self.rebuild_widget(number_of_plots, plot_stretching)

    def rebuild_widget(self, number_of_plots, plot_stretching):

        self.__number_of_plots = number_of_plots
        self.__plot_stretching = plot_stretching

        ids = []
        if self.__top_vbox.count() < self.__number_of_plots:
            ids = range(self.__top_vbox.count(), self.__number_of_plots)

        for i in ids:

            label = QLabel()
            label.setFont(self.__font)
            label.setAlignment(Qt.AlignCenter)

            # Create the mpl Figure and FigCanvas objects. 
            # 5x4 inches, 100 dots-per-inch
            #
            #dpi = 100
            #self.fig = Figure((5.0, 4.0), dpi=self.dpi)
            fig = pyplot.figure()
            canvas = FigureCanvas(fig)
            canvas.setParent(self)
    
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
            tmp_vbox.addWidget(label)
            tmp_vbox.addWidget(canvas, 1)
            tmp_vbox.addWidget(mpl_toolbar)

            widget = QWidget()
            widget.setLayout(tmp_vbox)

            self.__plots.append((label, canvas, fig, axes, mpl_toolbar, widget))

            self.__plot_infos.append((self.PLOT_TYPE_NONE, '', None, {}))

            self.__top_vbox.addWidget(widget)

        for i in xrange(self.__number_of_plots):

            stretch = 0
            if plot_stretching != None:
                stretch = plot_stretching[i]
                self.__top_vbox.setStretch(i, stretch)
            else:
                self.__top_vbox.setStretch(i, 1)

            plot = self.__plots[i]
            label, canvas, fig, axes, mpl_toolbar, widget = plot
            widget.show()

        for i in xrange(self.__number_of_plots, self.__top_vbox.count()):

            plot = self.__plots[i]
            label, canvas, fig, axes, mpl_toolbar, widget = plot
            widget.hide()

        if self.__show_menu:
            menubar = QMenuBar()
            save_menu = menubar.addMenu("&Save")
    
            menu_items = []
            for i,plot_info in enumerate(self.__plot_infos):
                plot_caption = plot_info[1]
                menu_name = "Save &plot '%s' [%d]" % (plot_caption, i+1)
                if len(plot_caption) == 0:
                    menu_name = "Save &plot #%d" % (i+1)
                save_plot_action = self.__make_action(menu_name,
                                                     shortcut="Ctrl+P",
                                                     slot=functools.partial(self.on_save_plot, i))
                menu_items.append(save_plot_action)
            menu_items.append(None)
            save_all_action = self.__make_action("Save &all plots",
                shortcut="Ctrl+A", slot=self.on_save_all_plots)
            menu_items.append(save_all_action)
    
            self.__add_actions(save_menu, menu_items)
    
            self.layout().setMenuBar(menubar)

    def __add_actions(self, target, actions):
        for action in actions:
            if action is None:
                target.addSeparator()
            else:
                target.addAction(action)

    def __make_action( self, text, slot=None, shortcut=None, 
                        icon=None, tip=None, checkable=False, 
                        signal="triggered()"):
        action = QAction(text, self)
        if icon is not None:
            action.setIcon(QIcon(":/%s.png" % icon))
        if shortcut is not None:
            action.setShortcut(shortcut)
        if tip is not None:
            action.setToolTip(tip)
            action.setStatusTip(tip)
        if slot is not None:
            self.connect(action, SIGNAL(signal), slot)
        if checkable:
            action.setCheckable(True)
        return action
