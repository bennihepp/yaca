import sys, os, random
from PyQt4.QtCore import *
from PyQt4.QtGui import *

import numpy
import struct
import Image
import ImageChops

import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import pyplot

from gallery_window import GalleryWindow
from plot_window import PlotWindow
from gui_utils import CellPixmapFactory,CellFeatureTextFactory

from ..core import cluster



class ResultWindowNavigationToolbar( NavigationToolbar ):

    STATUS_LABEL_TEXT_SIZE = 10

    def __init__(self, canvas, parent, coordinates=True):
        NavigationToolbar.__init__( self, canvas, parent, False )
        self.__coordinates = coordinates
        self.__init_toolbar()

    def __init_toolbar(self):

        if self.__coordinates:
            self.__coordinate_label = QLabel( '', self )
            self.__coordinate_label.setAlignment(
                Qt.AlignRight | Qt.AlignVCenter
            )
        self.__mouseover_label = QLabel( '', self )
        self.__mouseover_label.setAlignment(
            Qt.AlignRight | Qt.AlignVCenter
        )
        self.__selection_label = QLabel( '', self )
        self.__selection_label.setAlignment(
            Qt.AlignRight | Qt.AlignVCenter
        )

        labels = [ self.__mouseover_label, self.__selection_label ]
        labels.append( self.__coordinate_label )

        font = QFont( self.__mouseover_label.font().family(), self.STATUS_LABEL_TEXT_SIZE )

        for label in labels:
            label.setFont( font )

        vbox = QVBoxLayout()
        if self.__coordinates:
            vbox.addWidget( self.__coordinate_label )
        vbox.addWidget( self.__mouseover_label )
        vbox.addWidget( self.__selection_label )

        widget = QWidget()
        widget.setLayout( vbox )
        widget.setSizePolicy( QSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Minimum
        ) )

        action = self.addWidget( widget )
        action.setVisible( True )

    def set_message(self, s):
        NavigationToolbar.set_message( self, s )
        if self.__coordinates:
            values = s.strip().split()
            if len( values ) == 2:
                x,y = values
                self.__coordinate_label.setText( '%s    %s' % ( x, y ) )
            else:
                self.__coordinate_label.setText( s.strip() )

    def clear_coordinates(self):
        self.__coordinate_label.setText( '' )

    def set_coordinates(self, x, y):
        self.__coordinate_label.setText( 'x=%d, y=%d' % ( x, y ) )

    def clear_mouseover(self):
        self.__mouseover_label.setText( '' )

    def set_mouseover(self, index, group_policy, group_name):
        self.__mouseover_label.setText(
            '%s=%s    cell=#%d' % ( group_policy, group_name, index ) )

    def clear_selection(self):
        self.__selection_label.setText( 'No selection' )

    def set_selection(self, index, group_policy, group_name):
        if index >= 0:
            self.__selection_label.setText(
                'selected %s=%s    cell=#%d' % ( group_policy, group_name, index ) )
        else:
            self.__selection_label.setText(
                'selected %s=%s' % ( group_policy, group_name ) )



class ResultsWindow(QWidget):

    GROUP_POLICY_TREATMENT = 0
    GROUP_POLICY_CLUSTERING = 1
    DEFAULT_GROUP_POLICY = GROUP_POLICY_TREATMENT

    def __init__(self, pipeline, channelMapping, channelDescription, simple_ui=False, parent=None):

        self.__simple_ui = simple_ui

        self.pipeline = pipeline

        self.pdc = pipeline.pdc

        self.features = pipeline.nonControlFeatures

        #self.mahalFeatures = pipeline.nonControlTransformedFeatures
        #self.featureNames = pipeline.featureNames
        self.partition = pipeline.nonControlPartition
        if pipeline.nonControlClusters != None:
            self.number_of_clusters = pipeline.nonControlClusters.shape[0]
        else:
            self.number_of_clusters = -1

        self.channelMapping = channelMapping
        self.channelDescription = channelDescription

        self.group_policy = self.DEFAULT_GROUP_POLICY

        if self.group_policy == self.GROUP_POLICY_CLUSTERING:
            self.sorting_by_group = numpy.argsort( self.partition )
        elif self.group_policy == self.GROUP_POLICY_TREATMENT:
            self.sorting_by_group = numpy.argsort( self.features[ : , self.pdc.objTreatmentFeatureId ] )

        QWidget.__init__(self, parent)
        self.setWindowTitle('Results')

        self.build_widget()

        featureDescription = dict( self.pdc.objFeatureIds )

        #self.mahalFeatureIdOffset = max( featureDescription.values() ) + 1
        #for i in xrange( self.mahalFeatures.shape[ 1 ] ):
        #    featureDescription[ 'mahal_' + self.featureNames[ i ] ] = i + self.mahalFeatureIdOffset

        self.gallery = GalleryWindow( featureDescription, self.channelMapping, self.channelDescription )
        self.__plot_window = PlotWindow()

        self.histogram = None
        self.barplot = None

        self.picked_index = -1
        self.picked_group = -1

        self.last_group = None
        self.last_index = None

        self.__button_pressed_pos = None

        self.on_draw()


    def closeEvent(self, ce):

        self.gallery.close()

        if self.histogram:
            self.histogram.close()

        try: self.__plot_window.close();
        except: pass
        try: self.clustering_plot_window.close()
        except: pass

        ce.accept()

    """def save_plot(self):
        file_choices = "PNG (*.png)|*.png"

        path = unicode(QFileDialog.getSaveFileName(self, 
                        'Save file', '', 
                        file_choices))
        if path:
            self.canvas.print_figure(path, dpi=self.dpi)
            self.statusBar().showMessage('Saved to %s' % path, 2000)"""


    def update_selection(self, groupId, focusId=-1, clicked=True, button=-1):

        """if self.group_policy == self.GROUP_POLICY_TREATMENT:

            if self.pipeline.nonControlPartition != None:
                self.on_draw_treatment_to_cluster_distribution(
                    groupId,
                    clicked and ( button == 3 )
                )

        else:

            self.on_draw_intra_cluster_distribution( groupId, clicked and button == 3 )"""

        if ( clicked and button == 3 ) or self.__plot_window.isVisible():
            self.on_show_plots( groupId )

        #self.on_draw_histogram( groupId, clicked and button == 3 )

        if clicked and ( button == 1 or self.gallery.isVisible() ):

            self.on_view_cells( groupId, focusId )


    def on_view_cells(self, groupId=-1, focusId=-1):

        if groupId < 0:
            groupId = self.picked_group

        selectionMask = self.get_id_mask_for_group( groupId )

        selectionIds = self.pdc.objFeatures[ : , self.pdc.objObjectFeatureId ][ selectionMask ]

        featureFactory = CellFeatureTextFactory( self.pdc, self.pipeline.nonControlCellMask )
        pixmapFactory = CellPixmapFactory( self.pdc, self.channelMapping, self.pipeline.nonControlCellMask )

        self.gallery.on_selection_changed(focusId, selectionIds, pixmapFactory, featureFactory)
        self.gallery.update_caption(
            'Showing cells from %s %s' % ( self.get_group_policy_name(), self.get_group_name() )
        )
        self.gallery.show()


    def on_show_plots(self, groupId=-1):

        self.on_draw_plots( groupId, True )


    def on_dissolve_cluster(self):

        groupId = self.picked_group

        self.pipeline.dissolve_cluster( groupId )

        self.on_group_policy_changed( self.GROUP_POLICY_CLUSTERING )


    def on_selection_changed(self, groupId, index=-1, clicked=True, button=-1):

        if groupId >= 0:
            self.mpl_toolbar.set_selection(
                index,
                self.get_group_policy_name(),
                self.get_group_name( groupId )
            )
        else:
            self.mpl_toolbar.clear_selection()

        self.group_combo.setCurrentIndex( groupId )

        self.picked_index = index
        self.picked_group = groupId
        if groupId < 0:
            self.histogram_button.setEnabled( False )
            self.view_cells_button.setEnabled( False )
            self.show_plots_button.setEnabled( False )
            self.dissolve_cluster_button.setEnabled( False )
            self.gallery.close()
            if self.histogram:
                self.histogram.close()
        else:
            self.histogram_button.setEnabled( True )
            self.view_cells_button.setEnabled( True )
            self.show_plots_button.setEnabled( True )
            self.dissolve_cluster_button.setEnabled( self.group_policy == self.GROUP_POLICY_CLUSTERING )
            self.update_selection( groupId, index, clicked, button )


    def on_merge_cluster(self):

        clusterId1 = self.picked_group
        clusterId2 = self.merge_cluster_combo.currentIndex()

        self.pipeline.merge_clusters( clusterId1, clusterId2 )

        self.on_group_policy_changed( self.GROUP_POLICY_CLUSTERING )

    def on_merge_cluster_combo_activated(self, groupId):
        if groupId >= 0:
            self.merge_cluster_button.setEnabled( True )
        else:
            self.merge_cluster_button.setEnabled( False )


    def on_group_combo_activated(self, groupId):
        self.on_selection_changed( groupId )


    def on_figure_leave(self, event):

        #self.mpl_toolbar.clear_coordinates()
        self.mpl_toolbar.clear_mouseover()

        #self.upper_statusbar1.clearMessage()
        #self.upper_statusbar2.clearMessage()

        if self.last_group != self.last_index or self.last_index != self.picked_index:

            if self.last_group != None:
                self.update_selection( self.picked_group, self.picked_index, False )

            self.last_group = None
            self.last_index = None

    def on_axes_leave(self, event):

        #self.mpl_toolbar.clear_coordinates()
        self.mpl_toolbar.clear_mouseover()

        #self.upper_statusbar1.clearMessage()
        #self.upper_statusbar2.clearMessage()

        if self.last_group != self.last_index or self.last_index != self.picked_index:

            if self.last_group != None:
                self.update_selection( self.picked_group, self.picked_index, False )

            self.last_group = None
            self.last_index = None


    def on_motion_notify(self, event):
        # The event received here is of the type
        # matplotlib.backend_bases.MouseEvent
        #
        # It carries lots of information, of which we're using
        # only a small amount here.

        mx = event.xdata
        my = event.ydata

        if mx != None and my != None:

            x_lim = self.axes.get_xlim()
            y_lim = self.axes.get_ylim()
            x_weight = 1 / ( x_lim[1] - x_lim[0] )
            y_weight = 1 / ( y_lim[1] - y_lim[0] )
            d = ( x_weight * ( self.x_data - mx ) ) ** 2 + ( y_weight * ( self.y_data - my ) ) ** 2
            min_i = -1
            min_d = 4 * x_weight
            for i in xrange(d.shape[0]):
                if d[i] < min_d:
                    min_i = i
                    min_d = d[i]
            index = min_i
    
            if index >= 0:
    
                groupId = self.get_group_id( index )
    
                self.mpl_toolbar.set_mouseover(
                    index,
                    self.get_group_policy_name(),
                    self.get_group_name( groupId)
                )

                #self.upper_statusbar1.showMessage(
                #    '%s=%s' % ( self.get_group_policy_name(), self.get_group_name( groupId) )
                #)
                #self.upper_statusbar2.showMessage(
                #    'cell ID=%d' % ( index )
                #)
    
                if groupId != self.last_group:

                    self.update_selection( groupId, index, False )
                    self.last_group = groupId
                    self.last_index = index

            else:

                if self.last_group != None:
                    self.update_selection( self.picked_group, self.picked_index, False )
                    self.last_group = None
                    self.last_index = None

                #self.mpl_toolbar.clear_coordinates()
                self.mpl_toolbar.clear_mouseover()

                #self.upper_statusbar1.clearMessage()
                #self.upper_statusbar2.clearMessage()


    def on_button_press(self, event):

        self.__button_pressed_pos = ( event.x, event.y )

    def on_button_release(self, event):
        # The event received here is of the type
        # matplotlib.backend_bases.PickEvent
        #
        # It carries lots of information, of which we're using
        # only a small amount here.

        if self.__button_pressed_pos == None:
            return
        if  self.__button_pressed_pos != ( event.x, event.y ):
            self.__button_pressed_pos = None
            return

        self.__button_pressed_pos = None

        self.on_button_clicked( event )

    def on_button_clicked(self, event):

        mx = event.xdata
        my = event.ydata

        if mx != None and my != None:

            x_lim = self.axes.get_xlim()
            y_lim = self.axes.get_ylim()
            x_weight = 1 / ( x_lim[1] - x_lim[0] )
            y_weight = 1 / ( y_lim[1] - y_lim[0] )
            d = ( x_weight * ( self.x_data - mx ) ) ** 2 + ( y_weight * ( self.y_data - my ) ) ** 2
            min_i = -1
            min_d = 4 * x_weight
            for i in xrange(d.shape[0]):
                if d[i] < min_d:
                    min_i = i
                    min_d = d[i]
            index = min_i
    
            if index >= 0:
    
                focusId = index
                groupId = self.get_group_id( index )
        
                self.picked_group = groupId
                self.picked_index = focusId

                self.mpl_toolbar.set_selection(
                    self.picked_index,
                    self.get_group_policy_name(),
                    self.get_group_name()
                )
                self.statusBarLabel.setText( "You selected %s '%s' and cell #%d" % \
                                    ( self.get_group_policy_name(),
                                      self.get_group_name(),
                                      self.picked_index ) )
        
                self.on_selection_changed( groupId, focusId, True, event.button )
    
            else:
    
                self.mpl_toolbar.clear_selection()
                self.statusBar.showMessage( 'invalid selection', 2000 )
                self.statusBarLabel.setText( 'no %s selected' % self.get_group_policy_name() )

                self.picked_group = None
                self.picked_index = None

                self.on_selection_changed( -1, -1, True, event.button )


    def on_draw(self):
        # Redraws the figure

        self.axes.clear()
        colors = ['b','g','c','m','y','r']
        symbols = ['s','+','*','d']
        markers = []
        for s in symbols:
            for c in colors:
                markers.append(s+c)
        picked_symbol = 'rp'
        if len(self.x_data_by_group) == len(self.y_data_by_group):
            for i in xrange(len(self.x_data_by_group)):
                #import pdb; pdb.set_trace()
                if self.x_data_by_group[i].shape != self.y_data_by_group[i].shape:
                    raise Exception('inconsistent data')
                self.axes.plot(
                        self.x_data_by_group[i],
                        self.y_data_by_group[i],
                        markers[ i % ( len( markers ) ) ],
                        picker = 4,
                        label = '%d' % i
                )
            #if self.picked_index < 0:
            #    self.axes.plot(self.x_data[self.picked_index],
            #                   self.y_data[self.picked_index],
            #                   picked_symbol, label = 'selected')

        self.canvas.draw()



    def get_group_id(self, id):

        if id < 0:
            return -1

        if self.group_policy == self.GROUP_POLICY_CLUSTERING:
            return int( self.partition[ id ] )
        elif self.group_policy == self.GROUP_POLICY_TREATMENT:
            return int( self.features[ id , self.pdc.objTreatmentFeatureId ] )

    def get_group_name(self, groupId=None):

        if groupId == None or groupId < 0:
            groupId = self.picked_group

        if self.group_policy == self.GROUP_POLICY_TREATMENT:
            return self.pdc.treatments[ groupId ].name
        else:
            return str( groupId )

    def get_group_policy_name(self, policy=None):

        if policy == None:
            policy = self.group_policy

        if self.group_policy == self.GROUP_POLICY_TREATMENT:
            return 'treatment'
        else:
            return 'cluster'


    def get_id_mask_for_group(self, groupId):

        if groupId >= 0:

            if self.group_policy == self.GROUP_POLICY_CLUSTERING:
                return self.partition == groupId
            elif self.group_policy == self.GROUP_POLICY_TREATMENT:
                tr = self.pdc.treatments[ groupId ]
                return self.features[ : , self.pdc.objTreatmentFeatureId ] == tr.rowId

        else:
            mask = numpy.empty( ( self.features.shape[0], ), dtype=numpy.bool )
            mask[:] = True
            return mask


    def get_number_of_groups(self):

        if self.group_policy == self.GROUP_POLICY_CLUSTERING:
            return self.pipeline.nonControlClusters.shape[0]
            #return self.number_of_clusters
        elif self.group_policy == self.GROUP_POLICY_TREATMENT:
            return len( self.pdc.treatments )


    def group_data_by_group(self, values):

        groups = []

        for i in xrange( self.get_number_of_groups() ):

            selection_mask = self.get_id_mask_for_group( i )
            selection = values[ selection_mask ]
            groups.append( selection )

        return groups


    def select_data(self, feature_id):

        values = self.features[ : , feature_id ]
        groups = self.group_data_by_group( values )

        return values, groups

    def select_data_by_name(self, data_name):

        data_name = str( data_name )

        if data_name.startswith( 'feature ' ):
            feature_id = int( data_name[ len( 'feature ' ) : ] )
            values = self.features[ : , feature_id ]
            #print 'featureId=%d' % feature_id

        #elif data_name.startswith( 'mahal ' ):
        #    feature_id = int( data_name[ len( 'mahal ' ) : ] )
        #    values = self.mahalFeatures[ : , feature_id ]

        elif data_name == 'n':
            values = numpy.arange( self.features.shape[0] )
            #values = numpy.empty( self.features.shape[0] )
            #for i in xrange( values.shape[0] ):
            #    values[i] = i

        elif data_name == 'n_sorted':
            values = numpy.empty( self.features.shape[0] )
            for i in xrange( values.shape[0] ):
                values[ self.sorting_by_group[ i ] ] = i

        else:
            raise Exception( 'Unknown selection of data' )

        groups = self.group_data_by_group( values )

        return values, groups


    def load_x_data(self, index):
        name = self.x_combo.itemData( index ).toString()
        #print 'loading %s -> %d' % ( name, index )
        self.x_data,self.x_data_by_group = self.select_data_by_name(name)

    def load_y_data(self, index):
        name = self.y_combo.itemData( index ).toString()
        #print 'loading %s -> %d' % ( name, index )
        self.y_data,self.y_data_by_group = self.select_data_by_name(name)

    def on_x_combo_changed(self, index):
        self.load_x_data( index )
        self.on_draw()

    def on_y_combo_changed(self, index):
        self.load_y_data( index )
        self.on_draw()

    def on_histogram_changed(self, i):
        self.on_draw_histogram()


    def draw_cluster_population(self, plot_window, plot_index, caption, clusters, partition):
        # retrieve data for cluster population
        population = numpy.empty( ( clusters.shape[0], ) )
        for k in xrange( clusters.shape[0] ):
            partition_mask = partition[:] == k
            k_population = numpy.sum( partition_mask )
            population[ k ] = k_population
        plot_window.draw_barplot( plot_index, caption, population )

    def draw_inter_cluster_distances(self, plot_window, plot_index, caption, inter_cluster_distances):

            plot_window.draw_heatmap(
                plot_index,
                caption,
                inter_cluster_distances,
                interpolation='nearest'
            )

    def on_draw_clustering_plots(self, create=True):

        try:
            self.clustering_plot_window
        except:
            self.clustering_plot_window = None
        if self.clustering_plot_window == None and create:
            self.clustering_plot_window = PlotWindow( 3, ( 3, 2, 2 ) )
        if create:
            self.clustering_plot_window.show()

        if self.clustering_plot_window != None:

            if self.pipeline.nonControlPartition == None:
                self.clustering_plot_window.hide()

            if self.clustering_plot_window.isVisible():

                self.draw_inter_cluster_distances(
                    self.clustering_plot_window,
                    0,
                    'Pairwise distances between the clusters',
                    self.pipeline.nonControlInterClusterDistances
                )

                self.draw_cluster_population(
                    self.clustering_plot_window,
                    1,
                    'Population size of each cluster',
                    self.pipeline.nonControlClusters,
                    self.pipeline.nonControlPartition
                )

                self.draw_weights(
                    self.clustering_plot_window,
                    2,
                    'Features importance'
                )



    def draw_treatment_to_cluster_distribution(self, plot_window, plot_index, caption, groupId):

        values,groups = self.select_data( self.pdc.objTreatmentFeatureId )

        trMask = self.features[ :, self.pdc.objTreatmentFeatureId ] == groupId

        values = numpy.zeros( ( self.pipeline.nonControlClusters.shape[0], ) )
        for i in xrange( self.pipeline.nonControlClusters.shape[0] ):
            values[ i ] = numpy.sum( self.partition[ trMask ] == i )

        bin_labels = []
        bin_rescale = None

        for i in xrange( self.pipeline.nonControlClusters.shape[0] ):
            label = '%d' % i
            bin_labels.append( label )

        x = numpy.arange( self.pipeline.nonControlClusters.shape[0] )

        plot_window.draw_barplot(plot_index, caption, values, x, bin_labels )

    def draw_intra_cluster_distribution(self, plot_window, plot_index, caption, groupId):

        distances = self.pipeline.nonControlIntraClusterDistances

        mask = self.get_id_mask_for_group( groupId )
        distances = distances[ mask ]
        distances = numpy.sort( distances )

        bins = distances.shape[0] / 20

        plot_window.draw_histogram( plot_index, caption, distances, bins )

    def draw_cluster_to_treatment_distribution(self, plot_window, plot_index, caption, groupId):

        # retrieve data for treatment histogram
        values,groups = self.select_data( self.pdc.objTreatmentFeatureId )
        values = groups[ groupId ]
        bins = len( self.pdc.treatments )
        bin_labels = []
        bin_rescale = None
        for tr in self.pdc.treatments:
            tr_mask = self.features[ : , self.pdc.objTreatmentFeatureId ] == tr.rowId
            tr_obj_count = numpy.sum( tr_mask )
            ratio = numpy.sum( values[:] == tr.rowId )
            ratio /= float( tr_obj_count )
            ratio *= 100.0
            label = '%s\n%d%%' % ( tr.name, ratio )
            bin_labels.append( label )
        plot_window.draw_histogram( plot_index, caption, values, bins, bin_labels, bin_rescale )

    def draw_treatment_population(self, plot_window, plot_index, caption, groupId):

        # retrieve data for cluster population
        population = numpy.empty( len( self.pdc.treatments ) )
        x = numpy.arange( len( self.pdc.treatments ) )
        x_labels = []
        for tr in self.pdc.treatments:
            tr_mask = self.features[ : , self.pdc.objTreatmentFeatureId ] == tr.rowId
            tr_population = numpy.sum( tr_mask )
            population[ tr.rowId ] = tr_population
            x_labels.append( tr.name )
        plot_window.draw_barplot( plot_index, caption, population, x, x_labels, facecolor='blue' )

    def draw_cluster_similarities(self, plot_window, plot_index, caption, groupId):
    
        weights = None
        if self.pipeline.nonControlWeights != None:
            # retrieve data for cluster population
            #weights = self.pipeline.nonControlWeights[ groupId ]
            weights = self.pipeline.nonControlWeights
            if len( weights.shape ) > 1:
                weights = weights[ groupId ]

        clusters = self.pipeline.nonControlClusters
        partition = self.pipeline.nonControlPartition

        # remove the cluster centroid from the list of centroids
        centroid_mask = numpy.empty( ( clusters.shape[0], ), dtype=numpy.bool)
        centroid_mask[:] = True
        centroid_mask[ groupId ] = False
        clusters = clusters[ centroid_mask ]
        if weights != None and len( weights.shape ) > 1:
            weights = weights[ centroid_mask ]
        partition_mask = partition[:] == groupId
        # partition the cells of this cluster along
        # the remaining clusters
        points = self.pipeline.nonControlNormFeatures
        new_partition = cluster.find_nearest_cluster(
                                    points[ partition_mask ],
                                    clusters,
                                    weights,
                                    2
        )
        # reassign partition numbers
        reassign_mask = new_partition[:] >= groupId
        new_partition[ reassign_mask ] += 1
        values = numpy.zeros( ( self.pipeline.nonControlClusters.shape[0], ) )
        for k in xrange( values.shape[0] ):
            values[ k ] = numpy.sum( new_partition == k )
        x = numpy.arange( values.shape[0] )

        plot_window.draw_barplot( plot_index, caption, values, x, facecolor='orange' )


    def draw_weights(self, plot_window, plot_index, caption, groupId=-1):

        if self.pipeline.nonControlWeights != None:
            # retrieve data for cluster population
            #weights = self.pipeline.nonControlWeights[ groupId ]
            weights = self.pipeline.nonControlWeights
            #if len( weights.shape ) > 1:
            #    if groupId < 0:
            #        groupId = self.picked_group
            #        if groupId < 0:
            #            return
            #    weights = weights[ groupId ]
            weights = weights[0]
            plot_window.draw_barplot( plot_index, caption, weights, facecolor='purple' )

    def on_draw_plots(self, groupId=-1, create=True):

        #try:
        #    self.__plot_window
        #except:
        #    self.__plot_window = None
        #if self.__plot_window == None and create:
        #    self.__plot_window = PlotWindow()
        if create:
            self.__plot_window.show()

        if groupId < 0:
            groupId = self.picked_group

        if groupId < 0 and self.__plot_window != None:
            self.__plot_window.hide()

        if self.__plot_window != None and self.__plot_window.isVisible():

            #if groupId < 0:
            #    #self.statusBar.showMessage( 'You have to select a group first!', 2000 )
            #    return

            if self.pipeline.nonControlPartition == None:
                self.__plot_window.set_number_of_plots( 1 )
                # TODO
                #self.__plot_window = change_plot_window(
                #    self.__plot_window,
                #    1
                #)
            else:
                self.__plot_window.set_number_of_plots( 2, ( 3, 2 ) )
                # TODO
                #self.__plot_window = change_plot_window(
                #    self.__plot_window,
                #    2,
                #    ( 2, 1 ),
                #)

            if self.group_policy == self.GROUP_POLICY_TREATMENT:

                self.draw_treatment_population(
                    self.__plot_window,
                    0,
                    'Population size of each treatment',
                    groupId
                )

                if self.pipeline.nonControlPartition != None:

                    self.draw_treatment_to_cluster_distribution(
                        self.__plot_window,
                        1,
                        'Distribution of the selected treatment along the clusters',
                        groupId
                    )

            else:

                self.draw_cluster_to_treatment_distribution(
                    self.__plot_window,
                    0,
                    'Treatments',
                    groupId
                )

                if self.pipeline.nonControlPartition != None:

                    self.draw_cluster_similarities(
                        self.__plot_window,
                        1,
                        'Similarity to other clusters',
                        groupId
                    )

                    #self.draw_intra_cluster_distribution(
                    #    self.__plot_window,
                    #    2,
                    #    'Clusters',
                    #    groupId
                    #)

                    #self.draw_weights(
                    #    self.__plot_window,
                    #    2,
                    #    'Weights',
                    #    groupId
                    #)


    def on_draw_histogram(self, groupId=-1, create=True):

        try:
            self.histogram
        except:
            self.histogram = None
        if self.histogram == None and create:
            self.histogram = PlotWindow()
        if create:
            self.histogram.show()

        if groupId < 0:
            groupId = self.picked_group

        if self.histogram != None and self.histogram.isVisible():

            if groupId < 0:
                self.statusBar.showMessage( 'You have to select a group first!', 2000 )
            else:
                index = self.histogram_combo.currentIndex()
                name = str( self.histogram_combo.itemData( index ).toString() )
                values,groups = self.select_data( name )
                if name.startswith( 'feature ' ):
                    feature_id = int( name[ len( 'feature ' ) : ] )
                else:
                    feature_id = -1
                values = groups[ groupId ]
                bins = self.histogram_spinbox.value()
                if self.histogram == None:
                    self.histogram = PlotWindow()
                bin_labels = None
                bin_rescale = None
                if feature_id == self.pdc.objTreatmentFeatureId:
                    if bins == len( self.pdc.treatments ):
                        bin_labels = []
                        #bin_rescale = numpy.empty( ( bins, ) )
                        bin_rescale = None
                        for tr in self.pdc.treatments:
                            tr_mask = self.features[ : , self.pdc.objTreatmentFeatureId ] == tr.rowId
                            tr_obj_count = numpy.sum( tr_mask )
                            #bin_rescale[ tr.rowId ] = 100.0 / tr_obj_count
                            ratio = numpy.sum( values[:] == tr.rowId )
                            ratio /= float( tr_obj_count )
                            ratio *= 100.0
                            #ratio = values[ tr.rowId ] / float( tr_obj_count ) * 100.0
                            #ratio = values[ tr.rowId ]
                            #ratio = tr_obj_count
                            label = '%s\n%d%%' % ( tr.name, ratio )
                            bin_labels.append( label )
                self.histogram.draw_histogram( 0, values, bins, bin_labels, bin_rescale )
                self.histogram.show()


    def on_number_of_clusters_changed(self, number_of_clusters):
        self.number_of_clusters = number_of_clusters

    def on_cluster_combo_changed(self, index):
        self.supercluster_index = index

    def on_supercluster_changed(self, index):
        self.supercluster_index = index

    def on_cluster_button(self):

        self.statusBar.showMessage( 'Running clustering...' )

        self.pipeline.run_clustering( self.supercluster_index, self.number_of_clusters)

        """#self.pipeline.run_clustering( self.number_of_clusters )

        findNumberOfClusters = bool( self.findNumberOfClusters_checkBox.isChecked() )

        if findNumberOfClusters:
            number_of_clusters, gaps, sk = self.pipeline.prepare_clustering( self.number_of_clusters )
            print 'best number of clusters: %d' % number_of_clusters
        else:
            number_of_clusters = self.number_of_clusters

        calculate_silhouette = bool( self.findNumberOfClusters_checkBox.isChecked() )
        calculate_silhouette = False

        if calculate_silhouette:
            n_offset = 2
        else:
            n_offset = number_of_clusters

        s = []
        p = []
        for n in xrange( n_offset, number_of_clusters + 1 ):
            print 'clustering with %d clusters...' % n
            self.pipeline.run_clustering( n, calculate_silhouette )
            partition = self.pipeline.nonControlPartition
            p.append( partition )
            if calculate_silhouette:
                silhouette = self.pipeline.nonControlSilhouette
                s.append( numpy.mean( silhouette ) )
                print s[ -1 ]

        if calculate_silhouette:
            s = numpy.array( s )
            n = numpy.argmax( s ) + n_offset
        else:
            n = 0

        self.partition = p[ n ]
        self.number_of_clusters = n + n_offset"""

        self.partition = self.pipeline.nonControlPartition

        self.features = self.pipeline.nonControlFeatures
        #self.mahalFeatures = self.pipeline.nonControlTransformedFeatures
        #self.featureNames = self.pipeline.featureNames
        #self.partition = self.pipeline.nonControlPartition
        #self.number_of_clusters = self.pipeline.nonControlClusters.shape[0]

        self.statusBar.showMessage( 'Finished clustering!', 2000 )

        #self.supercluster_combo.setEnabled( True )
        #self.supercluster_label.setEnabled( True )

        self.clusteringRadiobutton.setEnabled( True )
        self.clusteringRadiobutton.setChecked( True )

        self.group_combo.clear()
        for i in xrange( self.get_number_of_groups() ):
            self.group_combo.addItem( str( i ), i )

        self.group_combo.setCurrentIndex( -1 )

        self.on_group_policy_changed( self.GROUP_POLICY_CLUSTERING )

        """if findNumberOfClusters:
            self.barplot = PlotWindow()
            self.barplot.draw_barplot( gaps, range( 1, len( gaps ) + 1 ), yerr=sk )
            self.barplot.show()

        if calculate_silhouette:
            self.barplot = PlotWindow()
            #silhouette = self.pipeline.nonControlSilhouette
            #print silhouette
            #print silhouette.shape
            #self.barplot.draw_barplot( silhouette )
            self.barplot.draw_barplot( s )
            self.barplot.show()"""


    def on_group_policy_changed(self, group_policy):

        self.group_policy = group_policy

        self.picked_group = self.get_group_id( self.picked_index )

        self.mpl_toolbar.set_selection(
            self.picked_index,
            self.get_group_policy_name(),
            self.get_group_name()
        )

        if self.group_policy == self.GROUP_POLICY_CLUSTERING:
            self.sorting_by_group = numpy.argsort( self.partition )
        elif self.group_policy == self.GROUP_POLICY_TREATMENT:
            self.sorting_by_group = numpy.argsort( self.features[ : , self.pdc.objTreatmentFeatureId ] )

        self.group_combo.clear()
        for i in xrange( self.get_number_of_groups() ):
            name = str( i )
            if self.group_policy == self.GROUP_POLICY_TREATMENT:
                name = self.pdc.treatments[ i ].name
            self.group_combo.addItem( name, i )

        self.merge_cluster_combo.setEnabled( True )
        self.merge_cluster_combo.setCurrentIndex( -1 )
        if self.group_policy == self.GROUP_POLICY_CLUSTERING:
            self.merge_cluster_combo.clear()
            for i in xrange( self.get_number_of_groups() ):
                name = str( i )
                self.merge_cluster_combo.addItem( name, i )

        self.load_x_data( self.x_combo.currentIndex() )
        self.load_y_data( self.y_combo.currentIndex() )
        self.on_draw()

        if self.group_policy == self.GROUP_POLICY_CLUSTERING:
            self.on_draw_clustering_plots( True )

        self.on_draw_plots( -1, False )

        self.last_group = None
        self.last_index = None

        self.on_selection_changed( self.picked_group, self.picked_index, False )


    def on_view_all_cells(self):

        #if self.histogram != None and self.histogram.isVisible():
        #    self.on_draw_histogram()

        selectionMask = self.get_id_mask_for_group( -1 )

        selectionIds = self.pdc.objFeatures[ : , self.pdc.objObjectFeatureId ][ selectionMask ]

        featureFactory = CellFeatureTextFactory( self.pdc, self.pipeline.nonControlCellMask )
        pixmapFactory = CellPixmapFactory( self.pdc, self.channelMapping, self.pipeline.nonControlCellMask )

        self.gallery.on_selection_changed(-1, selectionIds, pixmapFactory, featureFactory)
        self.gallery.update_caption(
            'Showing all cells'
        )
        self.gallery.show()


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

        # Bind the 'pick' event for clicking on one of the bars
        #
        #self.canvas.mpl_connect( 'pick_event', self.on_pick )
        self.canvas.mpl_connect( 'button_press_event', self.on_button_press )
        self.canvas.mpl_connect( 'button_release_event', self.on_button_release )

        # Bind the 'motion_notify' event for moving the mouse
        self.canvas.mpl_connect( 'motion_notify_event', self.on_motion_notify )

        # Bind the 'axes_leave' event for leaving a figure with the mouse
        self.canvas.mpl_connect( 'axes_leave_event', self.on_axes_leave )

        # Bind the 'figure_leave' event for leaving a figure with the mouse
        self.canvas.mpl_connect( 'figure_leave_event', self.on_figure_leave )

        # Create the navigation toolbar, tied to the canvas
        #
        self.mpl_toolbar = ResultWindowNavigationToolbar(self.canvas, self)

        # Other GUI controls
        #

        treatmentRadiobutton = QRadioButton( 'Treatment' )
        clusteringRadiobutton = QRadioButton( 'Clustering' )
        self.buttonGroup = QButtonGroup()
        self.buttonGroup.addButton( treatmentRadiobutton, self.GROUP_POLICY_TREATMENT )
        self.buttonGroup.addButton( clusteringRadiobutton, self.GROUP_POLICY_CLUSTERING)
        for button in self.buttonGroup.buttons():
            if self.buttonGroup.id( button ) == self.group_policy:
                button.setChecked( True )
            else:
                button.setChecked( False )
        self.connect( self.buttonGroup, SIGNAL('buttonClicked(int)'), self.on_group_policy_changed )


        self.group_combo = QComboBox()

        for i in xrange( self.get_number_of_groups() ):
            self.group_combo.addItem( str( i ), i )

        self.group_combo.setCurrentIndex( -1 )

        self.connect( self.group_combo, SIGNAL('activated(int)'), self.on_group_combo_activated )

        hbox = QHBoxLayout()

        hbox2 = QHBoxLayout()
        hbox2.addWidget( QLabel( 'Select group:' ) )
        hbox2.addWidget( self.group_combo, 1 )

        self.view_cells_button = QPushButton( 'View cells' )
        self.view_cells_button.setEnabled( False )
        self.connect( self.view_cells_button, SIGNAL('clicked()'), self.on_view_cells )

        self.show_plots_button = QPushButton( 'Show plots' )
        self.show_plots_button.setEnabled( False )
        self.connect( self.show_plots_button, SIGNAL('clicked()'), self.on_show_plots )

        self.dissolve_cluster_button = QPushButton( 'Dissolve cluster' )
        self.dissolve_cluster_button.setEnabled( False )
        self.connect( self.dissolve_cluster_button, SIGNAL('clicked()'), self.on_dissolve_cluster )

        self.merge_cluster_button = QPushButton( 'Merge with cluster' )
        self.merge_cluster_button.setEnabled( False )
        self.connect( self.merge_cluster_button, SIGNAL('clicked()'), self.on_merge_cluster )

        self.merge_cluster_combo = QComboBox()
        self.merge_cluster_combo.setCurrentIndex( -1 )
        self.merge_cluster_combo.setEnabled( False )
        self.connect( self.merge_cluster_combo, SIGNAL('activated(int)'), self.on_merge_cluster_combo_activated )

        hbox3 = QHBoxLayout()
        hbox3.addWidget( self.merge_cluster_button )
        hbox3.addWidget( self.merge_cluster_combo )

        vbox = QVBoxLayout()
        vbox.addLayout( hbox2 )
        vbox.addWidget( self.view_cells_button )
        vbox.addWidget( self.show_plots_button )
        vbox.addWidget( self.dissolve_cluster_button )
        vbox.addLayout( hbox3 )

        hbox.addLayout( vbox )

        hbox2 = QHBoxLayout()
        hbox2.addWidget( treatmentRadiobutton )
        hbox2.addWidget( clusteringRadiobutton )
        groupBox = QGroupBox( 'Grouping policy' )
        groupBox.setLayout( hbox2 )

        hbox.addWidget( groupBox )


        self.number_of_clusters = len( self.pdc.treatments )

        self.supercluster_index = 0

        self.cluster_spinbox = QSpinBox()
        self.cluster_spinbox.setRange( 1, 100 )
        self.cluster_spinbox.setValue( self.number_of_clusters )
        self.connect( self.cluster_spinbox, SIGNAL('valueChanged(int)'), self.on_number_of_clusters_changed )

        cluster_button = QPushButton( 'Run clustering' )
        self.connect( cluster_button, SIGNAL('clicked()'), self.on_cluster_button )

        hbox2 = QHBoxLayout()
        hbox2.addWidget( QLabel( 'Number of clusters:' ) )
        hbox2.addWidget( self.cluster_spinbox, 1 )
        #hbox2.addWidget( cluster_button, 1 )

        self.supercluster_label = QLabel( 'Select feature set' )
        self.supercluster_combo = QComboBox()
        for i in xrange( len( self.pipeline.clusterConfiguration ) ):
            name,config = self.pipeline.clusterConfiguration[ i ]
            self.supercluster_combo.addItem( name, i )
        self.supercluster_combo.setCurrentIndex( 0 )
        self.supercluster_index = 0
        self.connect( self.supercluster_combo, SIGNAL('currentIndexChanged(int)'), self.on_supercluster_changed )
        hbox3 = QHBoxLayout()
        hbox3.addWidget( self.supercluster_label )
        hbox3.addWidget( self.supercluster_combo )


        vbox = QVBoxLayout()
        vbox.addLayout( hbox2 )
        if not self.__simple_ui:
            vbox.addLayout( hbox3 )
        vbox.addWidget( cluster_button )
        groupBox = QGroupBox( 'Clustering' )
        groupBox.setLayout( vbox )
        hbox.addWidget( groupBox )


        groupBox1 = QGroupBox( 'Grouping' )
        groupBox1.setLayout( hbox )


        self.clusteringRadiobutton = clusteringRadiobutton
        clusteringRadiobutton.setEnabled( False )


        self.picked_index = -1
        self.x_data_by_group = []

        self.y_data_by_group = []

        self.x_combo = QComboBox()
        self.y_combo = QComboBox()

        self.x_combo.addItem('n','n')
        self.y_combo.addItem('n','n')

        self.x_combo.addItem('n sorted by group','n_sorted')
        self.y_combo.addItem('n sorted by group','n_sorted')

        #for i in xrange(self.mahalFeatures.shape[1]):
        #    self.x_combo.addItem('mahalanobis_%s' % self.featureNames[i],'mahal %d' % i)
        #    self.y_combo.addItem('mahalanobis_%s' % self.featureNames[i],'mahal %d' % i)

        keys = self.pdc.objFeatureIds.keys()
        keys.sort()
        for name in keys:
            featureId = self.pdc.objFeatureIds[ name ]
            self.x_combo.addItem(name,'feature %d' % featureId)
            self.y_combo.addItem(name,'feature %d' % featureId)

        self.connect(self.x_combo, SIGNAL('currentIndexChanged(int)'), self.on_x_combo_changed)
        self.connect(self.y_combo, SIGNAL('currentIndexChanged(int)'), self.on_y_combo_changed)
        self.x_combo.setCurrentIndex(1)
        self.y_combo.setCurrentIndex(2)


        hbox1 = QHBoxLayout()
        hbox1.addWidget(QLabel('x-Axis:'))
        hbox1.addWidget(self.x_combo, 1)
        hbox2 = QHBoxLayout()
        hbox2.addWidget(QLabel('y-Axis:'))
        hbox2.addWidget(self.y_combo, 1)

        vbox = QVBoxLayout()
        vbox.addLayout( hbox1 )
        vbox.addLayout( hbox2 )

        groupBox2 = QGroupBox( 'Axes' )
        groupBox2.setLayout( vbox )


        self.histogram_combo = QComboBox()

        self.histogram_combo.addItem('n','n')

        self.histogram_combo.addItem('n sorted by group','n_sorted')

        #for i in xrange(self.mahalFeatures.shape[1]):
        #    self.histogram_combo.addItem('mahalanobis_%s' % self.featureNames[i],'mahal %d' % i)

        keys = self.pdc.objFeatureIds.keys()
        keys.sort()
        for name in keys:
            featureId = self.pdc.objFeatureIds[ name ]
            self.histogram_combo.addItem(name,'feature %d' % featureId)
            if featureId == self.pdc.objTreatmentFeatureId:
                self.histogram_combo.setCurrentIndex( self.histogram_combo.count() - 1 )

        self.connect( self.histogram_combo, SIGNAL('currentIndexChanged(int)'), self.on_histogram_changed )

        self.histogram_button = QPushButton( 'Draw' )
        self.histogram_button.setEnabled( False )
        self.connect( self.histogram_button, SIGNAL('clicked()'), self.on_draw_histogram )

        self.histogram_spinbox = QSpinBox()
        self.histogram_spinbox.setRange( 1, 100 )
        self.histogram_spinbox.setValue( len( self.pdc.treatments ) )
        self.connect( self.histogram_spinbox, SIGNAL('valueChanged(int)'), self.on_histogram_changed )

        hbox = QHBoxLayout()
        hbox.addWidget( self.histogram_combo, 1 )
        hbox.addWidget( QLabel( 'Bins:' ) )
        hbox.addWidget( self.histogram_spinbox )
        hbox.addWidget( self.histogram_button )
        groupBox3 = QGroupBox( 'Histogram' )
        groupBox3.setLayout( hbox )





        """self.number_of_clusters = len( self.pdc.treatments )

        self.cluster_spinbox = QSpinBox()
        self.cluster_spinbox.setRange( 1, 100 )
        self.cluster_spinbox.setValue( self.number_of_clusters )
        self.connect( self.cluster_spinbox, SIGNAL('valueChanged(int)'), self.on_number_of_clusters_changed )

        #self.cluster_combo = QComboBox()
        #for id in xrange( len( self.pipeline.clusterConfiguration ) ):
        #    name,featureNames = self.pipeline.clusterConfiguration[ id ]
        #    self.cluster_combo.addItem( name, id )
        #self.connect( self.cluster_combo, SIGNAL('currentIndexChanged(int)'), self.on_cluster_combo_changed )
        #self.cluster_combo.setCurrentIndex( -1 )
        #self.cluster_combo.setCurrentIndex( 0 )

        cluster_button = QPushButton( 'Run clustering' )
        self.connect( cluster_button, SIGNAL('clicked()'), self.on_cluster_button )

        #self.findNumberOfClusters_checkBox = QCheckBox( 'Determine number of clusters' )
        #self.findNumberOfClusters_checkBox.setChecked( False )

        hbox = QHBoxLayout()
        hbox.addWidget( QLabel( 'Number of clusters:' ) )
        hbox.addWidget( self.cluster_spinbox, 1 )
        #hbox.addWidget( QLabel( 'SuperCluster:' ) )
        #hbox.addWidget( self.cluster_combo, 2 )
        #hbox.addWidget( self.findNumberOfClusters_checkBox )
        hbox.addWidget( cluster_button, 1 )
        groupBox4 = QGroupBox( 'Clustering' )
        groupBox4.setLayout( hbox )


        #self.supercluster_label = QLabel( 'Select SuperCluster' )

        #self.supercluster_combo = QComboBox()

        #for i in xrange( len( self.pipeline.clusterConfiguration ) ):
        #    name,config = self.pipeline.clusterConfiguration[ i ]
        #    self.supercluster_combo.addItem( name, i )

        #self.supercluster_combo.setCurrentIndex( 0 )
        self.supercluster_index = 0

        #self.connect( self.supercluster_combo, SIGNAL('currentIndexChanged(int)'), self.on_supercluster_changed )
        #self.supercluster_combo.setEnabled( False )
        #self.supercluster_label.setEnabled( False )

        #hbox = QHBoxLayout()
        #hbox.addWidget( QLabel( 'Select group:' ) )
        #hbox.addWidget( self.group_combo, 1 )
        #hbox.addWidget( self.supercluster_label )
        #hbox.addWidget( self.supercluster_combo, 1 )
        #groupBox5 = QGroupBox( 'Grouping' )
        #groupBox5.setLayout( hbox )"""


        #
        # Layout with box sizers
        #


        vbox1 = QVBoxLayout()

        vbox1.addWidget( groupBox1 )
        vbox1.addWidget( groupBox2 )
        vbox1.addWidget( groupBox3 )
        #vbox1.addWidget( groupBox4 )
        #vbox1.addWidget( groupBox5 )

        if self.__simple_ui:
            #groupBox2.hide()
            groupBox3.hide()

        self.statusBar = QStatusBar()
        self.statusBarLabel = QLabel()
        self.statusBar.addWidget( self.statusBarLabel )

        view_all_cells_button = QPushButton( 'View all cells' )
        self.connect( view_all_cells_button, SIGNAL('clicked()'), self.on_view_all_cells )


        #self.upper_statusbar1 = QStatusBar()
        #self.upper_statusbar1.setSizeGripEnabled( False )
        #self.upper_statusbar2 = QStatusBar()
        #self.upper_statusbar2.setSizeGripEnabled( False )

        #statusbar_vbox = QVBoxLayout()
        #statusbar_vbox.addWidget( self.upper_statusbar1 )
        #statusbar_vbox.addWidget( self.upper_statusbar2 )

        hbox = QHBoxLayout()
        hbox.addWidget( self.mpl_toolbar, 1 )
        #hbox.addLayout( statusbar_vbox, 1 )
        hbox.addWidget( view_all_cells_button )

        vbox = QVBoxLayout()
        vbox.addWidget(self.canvas, 1)
        vbox.addLayout( hbox )
        vbox.addLayout(vbox1)
        vbox.addWidget(self.statusBar)

        self.setLayout(vbox)
