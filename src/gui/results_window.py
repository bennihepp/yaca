# -*- coding: utf-8 -*-

"""
results_window.py -- Visualization of data and results.
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
import struct
import Image
import ImageChops

import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib import pyplot

import print_engine
from gallery_window import GalleryWindow
from multipage_window import MultiPageWindow
from plot_window import PlotWindow, MultiPagePlotWindow
from cluster_profiles_window import ClusterProfilesWindow
from treatment_comparison_window import TreatmentComparisonWindow
from gui_utils import CellPixmapFactory,CellFeatureTextFactory
from scripting_window import ScriptingWindow

from cell_mask_filter_widget import CellMaskFilterWidget

from ..core import cluster, grouping, pipeline
#from ..core.command_interpreter import ICmd

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

class ResultsWindow(QMainWindow):

    GROUP_POLICY_TREATMENT = 0
    GROUP_POLICY_CLUSTERING = 1
    DEFAULT_GROUP_POLICY = GROUP_POLICY_TREATMENT

    def __init__(self, pipeline, channelMapping, channelDescription, simple_ui=False, parent=None):

        global_ns = globals()
        local_ns = locals().copy()
        local_ns.update({'plt' : pyplot, 'pdc' : pipeline.pdc,
                          'pl' : pipeline, 'pipeline' : pipeline, 'np' : numpy,
                          'plot' : PlotWindow, 'multiplot' : MultiPagePlotWindow,
                          'resultsWindow' : self})
        #self.icmd = ICmd(local_ns, global_ns)

        self.scripting_window = ScriptingWindow(local_ns, global_ns)
        self.scripting_window.show()

        self.__simple_ui = simple_ui

        self.pipeline = pipeline

        self.pdc = pipeline.pdc

        self.features = self.pdc.objFeatures[pipeline.nonControlCellMask]

        #self.mahalFeatures = pipeline.nonControlTransformedFeatures
        #self.featureNames = pipeline.featureNames
        self.partition = pipeline.partition
        if pipeline.clusters != None:
            self.number_of_clusters = pipeline.clusters.shape[0]
        else:
            self.number_of_clusters = -1

        self.channelMapping = channelMapping
        self.channelDescription = channelDescription

        self.group_policy = self.DEFAULT_GROUP_POLICY

        if self.group_policy == self.GROUP_POLICY_CLUSTERING:
            self.sorting_by_group = numpy.argsort( self.partition )
        elif self.group_policy == self.GROUP_POLICY_TREATMENT:
            self.sorting_by_group = numpy.argsort( self.features[ : , self.pdc.objTreatmentFeatureId ] )

        QMainWindow.__init__(self, parent)
        self.setWindowTitle('Results')

        featureDescription = dict( self.pdc.objFeatureIds )

        #self.mahalFeatureIdOffset = max( featureDescription.values() ) + 1
        #for i in xrange( self.mahalFeatures.shape[ 1 ] ):
        #    featureDescription[ 'mahal_' + self.featureNames[ i ] ] = i + self.mahalFeatureIdOffset

        self.gallery = GalleryWindow( pipeline, featureDescription, self.channelMapping, self.channelDescription )
        self.__plot_window = PlotWindow()

        self.histogram = None
        self.barplot = None

        self.picked_index = -1
        self.picked_group = -1

        self.last_group = None
        self.last_index = None

        self.prange = None

        self.__button_pressed_pos = None

        self.__pipeline_running = False
        self.__clustering_done = False

        self.__cell_mask_dict = {
            'All' : [ numpy.ones( ( len( self.pdc.objects ), ), dtype=bool ), 'All cells' ],
            'Non-control-like' : [ self.pipeline.nonControlCellMask, 'Non-control-like cells' ],
            'Control-like' : [ self.pipeline.controlCellMask, 'Control-like cells' ],
            'Valid' : [ self.pipeline.validCellMask, 'All valid cells' ],
            'Invalid' : [ numpy.invert( self.pipeline.validCellMask ), 'All invalid cells' ]
        }
        self.__default_cell_mask_name = 'Non-control-like'

        self.build_menu()
        self.build_widget()

        #self.on_draw_general_plots( True )

        self.on_draw()

        #distanceHeatmap = numpy.zeros((3, 3),  dtype=float)
        #distanceHeatmap[0,1] = distanceHeatmap[1,0] = 0.68
        #distanceHeatmap[0,2] = distanceHeatmap[2,0] = 0.34
        #distanceHeatmap[1,2] = distanceHeatmap[2,1] = 0.59
        #profileLabels = ["a","b","c"]
        ##profiles = numpy.zeros((3, 4),  dtype=float)
        ##self.cluster_profiles_window = ClusterProfilesWindow(
        ##    profileLabels,  profiles,  distanceHeatmap,  ( 1, 2 )
        ##)
        ##self.cluster_profiles_window.show()        cdm = distanceHeatmap.copy()
        #import hcluster
        #cdm = distanceHeatmap.copy()
        #cdm[numpy.identity(cdm.shape[0], dtype=bool)] = 0.0
        #cdm = hcluster.squareform(cdm)
        #Z = hcluster.linkage(cdm, 'average')
        #dendrogram_window = PlotWindow(show_toolbar=True)
        #custom_kwargs = {
            #'Z' : Z,
            #'labels' : profileLabels,
            #'title' : 'Dendrogram',
            #'xlabel' : 'Treatment',
            #'ylabel' : 'Distance'
        #}
        #dendrogram_window.draw_custom(
            #0, 'Dendrogram', print_engine.draw_dendrogram,
            #custom_kwargs=custom_kwargs, want_figure=False
        #)
        #dendrogram_window.show()


        """print 'valid cells:'
        pdc = pipeline.pdc
        for tr in pdc.treatments:
            tr_mask = pdc.objFeatures[ self.pipeline.validCellMask ][ :,  pdc.objTreatmentFeatureId ] == tr.index
            print '  %s -> %d' % ( tr.name, numpy.sum( tr_mask ) )

        print 'non-control-like cells:'
        pdc = pipeline.pdc
        for tr in pdc.treatments:
            tr_mask = pdc.objFeatures[ self.pipeline.nonControlCellMask ][ :,  pdc.objTreatmentFeatureId ] == tr.index
            print '  %s -> %d' % ( tr.name, numpy.sum( tr_mask ) )"""


    def closeEvent(self, ce):

        self.gallery.close()

        if self.histogram:
            self.histogram.close()

        try: self.scripting_window.close()
        except: pass
        try: self.__plot_window.close();
        except: pass
        try: self.clustering_plot_window.close()
        except: pass
        try: self.hclustering_plot_window.close()
        except: pass
        try: self.general_plot_window.close()
        except: pass
        try: self.cluster_profiles_window.close()
        except: pass
        try: self.treatment_comparison_window.close()
        except: pass
        try: self.dendrogram_window.close()
        except: pass
        #try: self.contour_plot_window.close()
        #except: pass

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

            if self.pipeline.partition != None:
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


    def on_view_cells(self, groupId=-1, focusId=-1, tabs=[]):

        if groupId < 0:
            groupId = self.picked_group

        selectionMask = self.get_full_id_mask_for_group( groupId )

        caption = 'Showing cells from %s %s' % ( self.get_group_policy_name(), self.get_group_name() )
        print 'groupId:', groupId
        print caption

        self.view_cells( selectionMask, focusId, caption, tabs )

    def view_cells(self, selectionMask, focusId, caption, tabs=[]):

        selectionIds = self.pdc.objFeatures[ : , self.pdc.objObjectFeatureId ][ selectionMask ]

        #featureFactory = CellFeatureTextFactory( self.pdc, self.pipeline.nonControlCellMask )
        #pixmapFactory = CellPixmapFactory( self.pdc, self.channelMapping, self.pipeline.nonControlCellMask )

        featureFactory = CellFeatureTextFactory( self.pdc )
        pixmapFactory = CellPixmapFactory( self.pdc, self.channelMapping )

        if focusId >= 0:
            focusObjId = self.features[ focusId, self.pdc.objObjectFeatureId ]
        else:
            focusObjId = -1

        print 'selectionIds:', selectionIds.shape

        featureDescription = dict( self.pdc.objFeatureIds )
        self.gallery.update_feature_description( featureDescription )

        self.gallery.on_selection_changed( focusObjId, selectionIds, pixmapFactory, featureFactory )

        self.gallery.update_caption( caption )
        self.gallery.update_tabs( tabs )
        self.gallery.show()


    def on_print_cluster_population(self, pdf_filename=None):

        if pdf_filename == None:

            file_choices = "Cluster population (*.pdf)"
    
            pdf_filename = unicode( QFileDialog.getSaveFileName(self, 
                            'Print cluster population', '', 
                            file_choices))

        if not pdf_filename:
            return

        print 'printing cluster population...'
        #pdf_filename = '/home/hepp/cluster_profiles.pdf'
        print_engine.print_clustering_plot( self.pipeline, pdf_filename )
        #print_engine.view_file( pdf_filename )
        print 'done'

    def on_print_cell_selection_plots(self, pdf_filename=None):

        if pdf_filename == None:

            file_choices = "Cell selection plots (*.pdf)"
    
            pdf_filename = unicode( QFileDialog.getSaveFileName(self, 
                            'Print cell selection plots', '', 
                            file_choices))

        if not pdf_filename:
            return

        print 'printing cell selection plots...'
        print_engine.print_cell_selection_plot(self.pipeline, pdf_filename)
        print 'done'

    def on_print_cell_population_and_penetrance(self, pdf_filename=None):

        if pdf_filename == None:

            file_choices = "Cell population and penetrance (*.pdf)"
    
            pdf_filename = unicode( QFileDialog.getSaveFileName(self, 
                            'Print cell population and penetrance plots', '', 
                            file_choices))

        if not pdf_filename:
            return

        print 'printing cell population and penetrance plots...'
        print_engine.print_cell_populations_and_penetrance(self.pipeline, pdf_filename)
        print 'done'

    def on_print_cluster_profiles(self, pdf_filename=None):

        if pdf_filename == None:

            file_choices = "Cluster profiles (*.pdf)"
    
            pdf_filename = unicode( QFileDialog.getSaveFileName(self, 
                            'Print cluster profiles', '', 
                            file_choices))

        if not pdf_filename:
            return

        print 'printing cluster profiles...'
        #pdf_filename = '/home/hepp/cluster_profiles.pdf'
        distanceHeatmap = self.pipeline.distanceHeatmapDict[self.selected_groups[0]]
        mask1 = numpy.isfinite(distanceHeatmap)[0]
        mask2 = numpy.isfinite(distanceHeatmap)[:,0]
        valid_mask = numpy.logical_or(mask1, mask2)
        clusterProfileLabels = self.pipeline.clusterProfileLabelsDict[self.selected_groups[0]]
        labels = list(itertools.compress(clusterProfileLabels, valid_mask))
        clusterProfiles = self.pipeline.clusterProfilesDict[self.selected_groups[0]][valid_mask]
        print_engine.print_cluster_profiles_and_heatmap(labels, clusterProfiles, distanceHeatmap, pdf_filename)
        #print_engine.view_file( pdf_filename )
        print 'done'


    def on_print_norm0_cluster_profiles(self, pdf_filename=None):

        if pdf_filename == None:

            file_choices = "Cluster profiles (*.pdf)"
    
            pdf_filename = unicode( QFileDialog.getSaveFileName(self, 
                            'Print normalized 0.0 cluster profiles', '', 
                            file_choices))

        if not pdf_filename:
            return

        print 'printing normalized 0.0 cluster profiles...'
        #pdf_filename = '/home/hepp/cluster_profiles.pdf'
        sum_of_min_profileHeatmap, l2_norm_profileHeatmap, treatments = \
            print_engine.print_cluster_profiles_and_heatmap(
                self.pipeline.pdc,
                self.pipeline.clusterProfilesDict[self.pipeline.clusterProfilesDict.keys()[0]],
                pdf_filename, True, 0.0
            )
        #print_engine.view_file( pdf_filename )
        print 'done'

        return sum_of_min_profileHeatmap, l2_norm_profileHeatmap

    def on_print_norm1_cluster_profiles(self, pdf_filename=None):

        if pdf_filename == None:

            file_choices = "Cluster profiles (*.pdf)"
    
            pdf_filename = unicode( QFileDialog.getSaveFileName(self, 
                            'Print normalized 0.1 cluster profiles', '', 
                            file_choices))

        if not pdf_filename:
            return

        print 'printing normalized 0.1 cluster profiles...'
        #pdf_filename = '/home/hepp/cluster_profiles.pdf'
        sum_of_min_profileHeatmap, l2_norm_profileHeatmap, treatments = \
            print_engine.print_cluster_profiles_and_heatmap( self.pipeline.pdc, self.pipeline.clusterProfiles, pdf_filename, True, 0.1 )
        #print_engine.view_file( pdf_filename )
        print 'done'

        return sum_of_min_profileHeatmap, l2_norm_profileHeatmap

    def on_print_norm2_cluster_profiles(self, pdf_filename=None):

        if pdf_filename == None:

            file_choices = "Cluster profiles (*.pdf)"
    
            pdf_filename = unicode( QFileDialog.getSaveFileName(self, 
                            'Print normalized 0.2 cluster profiles', '', 
                            file_choices))

        if not pdf_filename:
            return

        print 'printing normalized 0.2 cluster profiles...'
        #pdf_filename = '/home/hepp/cluster_profiles.pdf'
        sum_of_min_profileHeatmap, l2_norm_profileHeatmap, treatments = \
            print_engine.print_cluster_profiles_and_heatmap( self.pipeline.pdc, self.pipeline.clusterProfiles, pdf_filename, True, 0.2 )
        #print_engine.view_file( pdf_filename )
        print 'done'

        return sum_of_min_profileHeatmap, l2_norm_profileHeatmap

    def on_print_norm3_cluster_profiles(self, pdf_filename=None):

        if pdf_filename == None:

            file_choices = "Cluster profiles (*.pdf)"
    
            pdf_filename = unicode( QFileDialog.getSaveFileName(self, 
                            'Print normalized 0.3 cluster profiles', '', 
                            file_choices))

        if not pdf_filename:
            return

        print 'printing normalized 0.3 cluster profiles...'
        #pdf_filename = '/home/hepp/cluster_profiles.pdf'
        sum_of_min_profileHeatmap, l2_norm_profileHeatmap, treatments = \
            print_engine.print_cluster_profiles_and_heatmap( self.pipeline.pdc, self.pipeline.clusterProfiles, pdf_filename, True, 0.3 )
        #print_engine.view_file( pdf_filename )
        print 'done'

        return sum_of_min_profileHeatmap, l2_norm_profileHeatmap

    def on_print_norm4_cluster_profiles(self, pdf_filename=None):

        if pdf_filename == None:

            file_choices = "Cluster profiles (*.pdf)"
    
            pdf_filename = unicode( QFileDialog.getSaveFileName(self, 
                            'Print normalized 0.4 cluster profiles', '', 
                            file_choices))
        if not pdf_filename:
            return

        print 'printing normalized 0.4 cluster profiles...'
        #pdf_filename = '/home/hepp/cluster_profiles.pdf'
        sum_of_min_profileHeatmap, l2_norm_profileHeatmap, treatments = \
            print_engine.print_cluster_profiles_and_heatmap( self.pipeline.pdc, self.pipeline.clusterProfiles, pdf_filename, True, 0.4 )
        #print_engine.view_file( pdf_filename )
        print 'done'

        return sum_of_min_profileHeatmap, l2_norm_profileHeatmap

    def on_print_all_cluster_profiles(self):

        file_choices = "Cluster profiles (*.pdf)"

        pdf_filename = unicode( QFileDialog.getSaveFileName(self, 
                        'Print all cluster profiles', '', 
                        file_choices))
        if not pdf_filename:
            return

        path, filename = os.path.split( pdf_filename )

        base = os.path.splitext( filename )[ 0 ]

        self.on_print_cluster_profiles( os.path.join( path, base + '.pdf' ) )

        sum_of_min_profileHeatmap0, l2_norm_profileHeatmap0 = self.on_print_norm0_cluster_profiles( os.path.join( path, 'norm_0.0_' + base + '.pdf' ) )
        self.write_profileHeatmapCSV( self.pipeline.pdc.treatments, sum_of_min_profileHeatmap0, os.path.join( path, 'sum_of_min_0.0_' + base + '.xls' ) )
        self.write_profileHeatmapCSV( self.pipeline.pdc.treatments, l2_norm_profileHeatmap0, os.path.join( path, 'l2_norm_0.0_' + base + '.xls' ) )

        sum_of_min_profileHeatmap1, l2_norm_profileHeatmap1 = self.on_print_norm1_cluster_profiles( os.path.join( path, 'norm_0.1_' + base + '.pdf' ) )
        self.write_profileHeatmapCSV( self.pipeline.pdc.treatments, sum_of_min_profileHeatmap1, os.path.join( path, 'sum_of_min_0.1_' + base + '.xls' ) )
        self.write_profileHeatmapCSV( self.pipeline.pdc.treatments, l2_norm_profileHeatmap1, os.path.join( path, 'l2_norm_0.1_' + base + '.xls' ) )

        sum_of_min_profileHeatmap2, l2_norm_profileHeatmap2 = self.on_print_norm2_cluster_profiles( os.path.join( path, 'norm_0.2_' + base + '.pdf' ) )
        self.write_profileHeatmapCSV( self.pipeline.pdc.treatments, sum_of_min_profileHeatmap2, os.path.join( path, 'sum_of_min_0.2_' + base + '.xls' ) )
        self.write_profileHeatmapCSV( self.pipeline.pdc.treatments, l2_norm_profileHeatmap2, os.path.join( path, 'l2_norm_0.2_' + base + '.xls' ) )

        sum_of_min_profileHeatmap3, l2_norm_profileHeatmap3 = self.on_print_norm3_cluster_profiles( os.path.join( path, 'norm_0.3_' + base + '.pdf' ) )
        self.write_profileHeatmapCSV( self.pipeline.pdc.treatments, sum_of_min_profileHeatmap3, os.path.join( path, 'sum_of_min_0.3_' + base + '.xls' ) )
        self.write_profileHeatmapCSV( self.pipeline.pdc.treatments, l2_norm_profileHeatmap3, os.path.join( path, 'l2_norm_0.3_' + base + '.xls' ) )

        sum_of_min_profileHeatmap4, l2_norm_profileHeatmap4 = self.on_print_norm4_cluster_profiles( os.path.join( path, 'norm_0.4_' + base + '.pdf' ) )
        self.write_profileHeatmapCSV( self.pipeline.pdc.treatments, sum_of_min_profileHeatmap4, os.path.join( path, 'sum_of_min_0.4_' + base + '.xls' ) )
        self.write_profileHeatmapCSV( self.pipeline.pdc.treatments, l2_norm_profileHeatmap4, os.path.join( path, 'l2_norm_0.4_' + base + '.xls' ) )

    def write_profileHeatmapCSV(self, treatments, heatmap, filename):

        f = open( filename, 'w' )

        f.write( '\t' )
        for i in xrange( len( treatments ) ):

            tr = treatments[ i ]

            str = '%s' % tr.name
            if i < len( treatments ) - 1:
                str += '\t'

            f.write( str )

        f.write( '\n' )

        for i in xrange( heatmap.shape[0] ):

            tr = treatments[ i ]
    
            str = '%s\t' % tr.name

            f.write( str )

            for j in xrange( heatmap.shape[1] ):

                str = '%f' % heatmap[ i, j ]
                if j < heatmap.shape[1] - 1:
                    str += '\t'

                f.write( str )
            f.write( '\n' )

        if len( treatments ) > 14:

            f.write( '\n' )

            f.write( '\t' )
            for i in xrange( len( treatments ) ):
    
                if i % 2 != 0:
                    continue

                tr = treatments[ i ]
    
                str = '%s' % tr.name
                if i < len( treatments ) - 1:
                    str += '\t'
    
                f.write( str )
    
            f.write( '\n' )

            f.write( 'self match\t' )
            for i in xrange( heatmap.shape[0] ):

                if i % 2 != 0:
                    continue

                str = '%f' % heatmap[ i, i+1 ]
                if i < heatmap.shape[0] - 2:
                    str += '\t'

                f.write( str )

            f.write( '\n' )

            f.write( 'median to others\t' )
            for i in xrange( heatmap.shape[0] ):
    
                if i % 2 != 0:
                    continue

                mask = numpy.ones( ( heatmap.shape[0], ), dtype=bool )
                mask[ i ] = False
                mask[ i+1 ] = False
                mask2 = numpy.any( numpy.isnan( heatmap[ [i,i+1] ] ), axis=0 )
                mask = numpy.logical_and( mask, numpy.invert( mask2 ) )

                med = numpy.median( heatmap[ [i,i+1] ][ :, mask ] )

                str = '%f' % med
                if i < heatmap.shape[0] - 2:
                    str += '\t'

                f.write( str )

            f.write( '\n' )

            f.write( 'mean to others\t' )
            for i in xrange( heatmap.shape[0] ):
    
                if i % 2 != 0:
                    continue

                mask = numpy.ones( ( heatmap.shape[0], ), dtype=bool )
                mask[ i ] = False
                mask[ i+1 ] = False
                mask2 = numpy.any( numpy.isnan( heatmap[ [i,i+1] ] ), axis=0 )
                mask = numpy.logical_and( mask, numpy.invert( mask2 ) )

                med = numpy.mean( heatmap[ [i,i+1] ][ :, mask ] )

                str = '%f' % med
                if i < heatmap.shape[0] - 2:
                    str += '\t'

                f.write( str )

            f.write( '\n' )

            f.write( 'min to others\t' )
            for i in xrange( heatmap.shape[0] ):
    
                if i % 2 != 0:
                    continue

                mask = numpy.ones( ( heatmap.shape[0], ), dtype=bool )
                mask[ i ] = False
                mask[ i+1 ] = False
                mask2 = numpy.any( numpy.isnan( heatmap[ [i,i+1] ] ), axis=0 )
                mask = numpy.logical_and( mask, numpy.invert( mask2 ) )

                try:
                    min = numpy.min( heatmap[ [i,i+1] ][ :, mask ] )
                except:
                    min = numpy.nan

                str = '%f' % min
                if i < heatmap.shape[0] - 2:
                    str += '\t'

                f.write( str )

            f.write( '\n' )

            f.write( 'max to others\t' )
            for i in xrange( heatmap.shape[0] ):
    
                if i % 2 != 0:
                    continue

                mask = numpy.ones( ( heatmap.shape[0], ), dtype=bool )
                mask[ i ] = False
                mask[ i+1 ] = False
                mask2 = numpy.any( numpy.isnan( heatmap[ [i,i+1] ] ), axis=0 )
                mask = numpy.logical_and( mask, numpy.invert( mask2 ) )

                try:
                    max = numpy.max( heatmap[ [i,i+1] ][ :, mask ] )
                except:
                    max = numpy.nan

                str = '%f' % max
                if i < heatmap.shape[0] - 2:
                    str += '\t'

                f.write( str )

            f.write( '\n' )

            f.write( 'min to others - self match\t' )
            for i in xrange( heatmap.shape[0] ):

                if i % 2 != 0:
                    continue

                self_match = heatmap[ i, i+1 ]

                mask = numpy.ones( ( heatmap.shape[0], ), dtype=bool )
                mask[ i ] = False
                mask[ i+1 ] = False
                mask2 = numpy.any( numpy.isnan( heatmap[ [i,i+1] ] ), axis=0 )
                mask = numpy.logical_and( mask, numpy.invert( mask2 ) )

                try:
                    min = numpy.min( heatmap[ [i,i+1] ][ :, mask ] )
                except:
                    min = numpy.nan

                str = '%f' % ( min - self_match )
                if i < heatmap.shape[0] - 2:
                    str += '\t'

                f.write( str )

            f.write( '\n' )

            f.write( 'self match - max to others\t' )
            for i in xrange( heatmap.shape[0] ):

                if i % 2 != 0:
                    continue

                self_match = heatmap[ i, i+1 ]

                mask = numpy.ones( ( heatmap.shape[0], ), dtype=bool )
                mask[ i ] = False
                mask[ i+1 ] = False
                mask2 = numpy.any( numpy.isnan( heatmap[ [i,i+1] ] ), axis=0 )
                mask = numpy.logical_and( mask, numpy.invert( mask2 ) )

                try:
                    max = numpy.min( heatmap[ [i,i+1] ][ :, mask ] )
                except:
                    max = numpy.nan

                str = '%f' % ( self_match - max )
                if i < heatmap.shape[0] - 2:
                    str += '\t'

                f.write( str )

            f.write( '\n' )

        f.close()


    def on_print_cells(self):

        printer = QPrinter( QPrinter.HighResolution | QPrinter.Color | QPrinter.Portrait | QPrinter.A4 )
        dialog = QPrintDialog( printer, self )
        dialog.setWindowTitle( 'Print cell galleries' )
        if dialog.exec_() != QDialog.Accepted:
            return

        print 'printing cell galleries...'

        pixmap_width = self.gallery.pixmap_width
        pixmap_height = self.gallery.pixmap_height

        PIXMAP_SPACING = 8

        num_of_rows = int( ( printer.height() + PIXMAP_SPACING ) / float( pixmap_height + PIXMAP_SPACING ) )
        num_of_columns = int( ( printer.width() + PIXMAP_SPACING ) / float( pixmap_width + PIXMAP_SPACING ) )

        painter = QPainter()
        painter.begin( printer )

        self.gallery.hide()

        for groupId in xrange( self.get_number_of_groups() ):

            selectionMask = self.get_full_id_mask_for_group( groupId )

            if selectionMask.any():

                selectionIds = self.pdc.objFeatures[ : , self.pdc.objObjectFeatureId ][ selectionMask ]
    
                featureFactory = CellFeatureTextFactory( self.pdc )
                pixmapFactory = CellPixmapFactory( self.pdc, self.channelMapping )
    
                caption = 'Showing %d cells from %s %s' % ( num_of_rows*num_of_columns, self.get_group_policy_name(), self.get_group_name( groupId ) )
    
                self.gallery.on_selection_changed( -1, selectionIds, pixmapFactory, featureFactory )
    
                self.gallery.on_print( printer, painter, caption )

                if groupId < self.get_number_of_groups() - 1:
                    printer.newPage()

                sys.stdout.write( '\r  %s %s ...' % ( self.get_group_policy_name(), self.get_group_name( groupId ) ) )
                sys.stdout.flush()

        sys.stdout.write( '\n' )
        sys.stdout.flush()

        painter.end()

        self.statusBar().showMessage( 'Finished printing cells galleries', 2000 )

        print 'done'


    def on_show_plots(self, groupId=-1):

        self.on_draw_plots( groupId, True )


    def on_dissolve_cluster(self):

        groupId = self.picked_group

        self.pipeline.dissolve_cluster( groupId )

        self.on_group_policy_changed( self.GROUP_POLICY_CLUSTERING )


    def on_selection_changed(self, groupId, index=-1, clicked=True, button=-1):

        if groupId >= 0:
            objectId = self.features[ index, self.pdc.objObjectFeatureId ]
            self.mpl_toolbar.set_selection(
                objectId,
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
            min_d = 0.0001
            for i in xrange(d.shape[0]):
                if d[i] < min_d:
                    min_i = i
                    min_d = d[i]
            index = min_i
    
            if index >= 0:
    
                groupId = self.get_group_id( index )

                objectId = self.features[ index, self.pdc.objObjectFeatureId ]

                self.mpl_toolbar.set_mouseover(
                    objectId,
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

            #print 'x,y: (%f,%f)' % ( event.x, event.y )
            #print 'mx,my: (%f,%f)' % ( event.xdata, event.ydata )

            x_lim = self.axes.get_xlim()
            y_lim = self.axes.get_ylim()
            #print 'lim: [ (%f,%f) (%f,%f) ]' % ( x_lim[0], x_lim[1], y_lim[0], y_lim[1] )
            x_weight = 1 / ( x_lim[1] - x_lim[0] )
            y_weight = 1 / ( y_lim[1] - y_lim[0] )
            d = ( x_weight * ( self.x_data - mx ) ) ** 2 + ( y_weight * ( self.y_data - my ) ) ** 2
            #print 'weight: (%f,%f)' % ( x_weight, y_weight )
            #print d
            min_i = -1
            #min_d = 4 * x_weight
            min_d = 0.0001
            for i in xrange(d.shape[0]):
                if d[i] < min_d:
                    min_i = i
                    min_d = d[i]
            index = min_i

            #print 'min_d=%f' % min_d

            if index >= 0:
    
                focusId = index
                groupId = self.get_group_id( index )

                self.picked_group = groupId
                self.picked_index = focusId

                objectId = self.features[ self.picked_index, self.pdc.objObjectFeatureId ]

                #print 'index=%d, focusId=%d, groupId=%d, objectId=%d' % (index,focusId,groupId, objectId)

                self.mpl_toolbar.set_selection(
                    objectId,
                    self.get_group_policy_name(),
                    self.get_group_name()
                )
                self.statusBarLabel.setText( "You selected %s '%s' and cell #%d" % \
                                    ( self.get_group_policy_name(),
                                      self.get_group_name(),
                                      objectId ) )
        
                self.on_selection_changed( groupId, focusId, True, event.button )
    
            else:
    
                self.mpl_toolbar.clear_selection()
                self.statusBar().showMessage( 'invalid selection', 2000 )
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


    def get_full_id_mask_for_group(self, groupId):

        mask = numpy.empty( ( self.pdc.objFeatures.shape[0], ), dtype=numpy.bool )
        mask[:] = False

        mask2 = self.get_id_mask_for_group( groupId )
        #print 'mask2:', numpy.sum( mask2 )

        if self.group_policy == self.GROUP_POLICY_CLUSTERING:
            mask[self.pipeline.clusterCellMask] = mask2
        elif self.group_policy == self.GROUP_POLICY_TREATMENT:
            mask[self.pipeline.nonControlCellMask] = mask2

        return mask

    def get_id_mask_for_group(self, groupId):

        if groupId >= 0:

            if self.group_policy == self.GROUP_POLICY_CLUSTERING:
                return self.partition == groupId
                #return self.partition == groupId
            elif self.group_policy == self.GROUP_POLICY_TREATMENT:
                tr = self.pdc.treatments[ groupId ]
                #print 'retrieving id mask for treatment %d: %s' % ( tr.index, tr.name )
                mask = self.features[ : , self.pdc.objTreatmentFeatureId ] == tr.index
                #print 'mask:', numpy.sum( mask )
                return mask

        else:
            mask = numpy.empty( ( self.pdc.objFeatures.shape[0], ), dtype=numpy.bool )
            mask[:] = True
            return mask


    def get_number_of_groups(self):

        if self.group_policy == self.GROUP_POLICY_CLUSTERING:
            return self.pipeline.clusters.shape[0]
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

        #elif data_name == 'mahal_dist':
        #    values = self.pipeline.nonControlMahalDist
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

    def draw_cluster_stddevs(self, plot_window, plot_index, caption, clusters, partition, points):
        # Compute standard-deviation vector for each cluster
        stddevs = numpy.empty( clusters.shape )
        for i in xrange( clusters.shape[ 0 ] ):
            partition_mask = partition[:] == i
            stddevs[ i ] = numpy.std( points[ partition_mask ], axis=0 )

        from ..core import distance

        stddev_m = numpy.empty( ( clusters.shape[0], clusters.shape[0] ) )
        dist_m = distance.minkowski_cdist( clusters, clusters )
        for i in xrange( clusters.shape[ 0 ] ):
            for j in xrange( clusters.shape[ 0 ] ):
                # compute the normalized distance vector of the clusters
                dist = dist_m[ i, j ]
                dvec = clusters[i] - clusters[j]
                dvec_norm = dvec / dist
                # compute projected standard deviations
                proj_stddev1 = abs( numpy.sum( dvec_norm * stddevs[i] ) )
                proj_stddev2 = abs( numpy.sum( dvec_norm * stddevs[j] ) )
                #proj_stddev = min( proj_stddev1, proj_stddev2 )
                proj_stddev = proj_stddev1 + proj_stddev2
                stddev_m[ i, j ] = proj_stddev

        plot_window.draw_heatmap( plot_index, caption, stddev_m, interpolation='nearest' )

    def draw_inter_cluster_distances(self, plot_window, plot_index, caption, inter_cluster_distances):

            plot_window.draw_heatmap(
                plot_index,
                caption,
                inter_cluster_distances,
                interpolation='nearest'
            )

    def draw_hcluster_dendrogram_custom(self, fig, axes, custom_data, bottom_shift=0.0, **mpl_kwargs):

        import matplotlib
        import matplotlib.pylab
        be = matplotlib.pylab.get_backend()
        matplotlib.pylab.switch_backend( 'Agg' )
        #artist = matplotlib.pylab.figure()
        Z = custom_data[:,:4]
        import hcluster
        hcluster.dendrogram(
            self.pipeline.Z[:,:4],
            30,
            'lastp'
        )
        artist = matplotlib.pylab.gca()
        print artist, id( artist)
        print artist.artists
        print artist.get_figure()
        print artist.get_axes()
        for p in artist.patches:
            axes.add_patch( p )
        for a in artist.artists:
            axes.add_artist( a )
        for c in artist.collections:
            axes.add_collection( c )
        for l in artist.lines:
            axes.add_line( l )
        for t in artist.tables:
            axes.add_table( t )
        matplotlib.pylab.switch_backend( be )

    def draw_hcluster_dendrogram(self, plot_window, plot_index, caption):

        plot_window.draw_custom( plot_index, caption, self.draw_hcluster_dendrogram_custom, self.pipeline.Z )

    def draw_hcluster_size(self, plot_window, plot_index, caption):

        x = numpy.arange( 1, self.pipeline.Z.shape[0] + 1 )[ ::-1 ]
        plot_window.draw_lineplot( plot_index, caption, x, self.pipeline.Z[:,3] )

    def draw_hcluster_dist(self, plot_window, plot_index, caption):

        x = numpy.arange( 1, self.pipeline.Z.shape[0] + 1 )[ ::-1 ]
        plot_window.draw_lineplot( plot_index, caption, x, self.pipeline.Z[:,2] )

    def draw_hcluster_Z(self, plot_window, plot_index, caption, Z_index):

        x = numpy.arange( 1, self.pipeline.Z.shape[0] + 1 )[ ::-1 ]
        plot_window.draw_lineplot( plot_index, caption, x, self.pipeline.Z[:,Z_index] )

    def on_draw_hcluster_plots( self, create=True ):

        try:
            self.hclustering_plot_window
        except:
            self.hclustering_plot_window = None
        if self.hclustering_plot_window == None and create:
            self.hclustering_plot_window = PlotWindow( 3, ( 3, 3, 3, 3, 3, 3 ) )
        if create:
            self.hclustering_plot_window.show()

        if self.hclustering_plot_window != None:

            if self.pipeline.partition == None:
                self.hclustering_plot_window.hide()

            if self.hclustering_plot_window.isVisible():

                #self.draw_hcluster_dendrogram(
                #    self.hclustering_plot_window,
                #    0,
                #    'HCluster dendrogram'
                #)

                #self.draw_inter_cluster_distances(
                #    self.hclustering_plot_window,
                #    0,
                #    'Pairwise distances between the clusters',
                #    self.pipeline.nonControlInterClusterDistances
                #)

                self.draw_hcluster_dist(
                    self.hclustering_plot_window,
                    0,
                    'Sample distances'
                )

                self.draw_hcluster_size(
                    self.hclustering_plot_window,
                    1,
                    'New cluster sizes'
                )

                self.draw_hcluster_Z(
                    self.hclustering_plot_window,
                    2,
                    'Number of big clusters',
                    5
                )
                max_ids = numpy.argsort( self.pipeline.Z[:,5] )[ :: -1 ]
                for i in xrange( 10 ):
                    j = self.pipeline.partition.shape[0] - max_ids[ i ]
                    print 'i=%d, num_of_big_clusters=%d' % ( j, self.pipeline.Z[ max_ids[ i ] , 5 ] )

                #self.draw_hcluster_Z(
                #    self.hclustering_plot_window,
                #    1,
                #    'Number of clusters',
                #    4
                #)

                #self.draw_hcluster_Z(
                #    self.hclustering_plot_window,
                #    2,
                #    'Error summed square',
                #    5
                #)

                """self.draw_hcluster_Z(
                    self.hclustering_plot_window,
                    4,
                    'HCluster cluster count',
                    6
                )"""

                """self.draw_hcluster_Z(
                    self.hclustering_plot_window,
                    4,
                    'Error summed square * Number of clusters',
                    7
                )"""


    def draw_cluster_profile_comparison_custom(self, fig, axes, custom_data, bottom_shift=0.0, **mpl_kwargs):

        profile1, profile2, x, x_labels = custom_data

        min_profile = numpy.min( [ profile1, profile2 ], axis=0 )

        axes.clear()

        if 'facecolor' in mpl_kwargs:
            del mpl_kwargs[ 'facecolor' ]
        if not mpl_kwargs.has_key( 'alpha' ): mpl_kwargs[ 'alpha' ] = 0.75
        if not mpl_kwargs.has_key( 'align' ): mpl_kwargs[ 'align' ] = 'center'

        axes.bar( x, min_profile, color='yellow', **mpl_kwargs )

        axes.bar( x, profile1 - min_profile, bottom=min_profile, color='green', **mpl_kwargs )

        axes.bar( x, profile2 - min_profile, bottom=min_profile, color='red', **mpl_kwargs )

        if x_labels != None:
            axes.set_xticks( x )
            x_labels = axes.set_xticklabels( x_labels, rotation='270' )
            if bottom_shift != None and fig.subplotpars.bottom < bottom_shift:
                fig.subplots_adjust( bottom=bottom_shift )

            axes.grid( True )

        return x_labels

    def draw_cluster_profile_comparison(self, plot_window, plot_index, caption, treatmentId1, treatmentId2):

        group = self.selected_groups[0]
        profile1 = self.pipeline.clusterProfilesDict[group][treatmentId1]
        profile2 = self.pipeline.clusterProfilesDict[group][treatmentId2]

        """trMask = self.features[ :, self.pdc.objTreatmentFeatureId ] == groupId

        print 'trMask: %d' % ( numpy.sum( trMask ) )
        values = numpy.zeros( ( self.pipeline.clusters.shape[0], ) )
        for i in xrange( self.pipeline.clusters.shape[0] ):
            values[ i ] = numpy.sum( self.partition[ trMask ] == i )
            print '%d: %d' % ( i, values[ i ] )

        print numpy.sum( values )"""

        x_labels = []

        for i in xrange( self.pipeline.clusters.shape[0] ):
            label = '%d' % i
            x_labels.append( label )

        x = numpy.arange( self.pipeline.clusters.shape[0] )

        plot_window.draw_custom(plot_index, caption, self.draw_cluster_profile_comparison_custom, [profile1, profile2, x, x_labels])


    def on_draw_cluster_profiles(self, treatmentId1=-1, treatmentId2=-1, create=True):

        try:
            self.cluster_profiles_window
        except:
            self.cluster_profiles_window = None
        if self.cluster_profiles_window == None and create:
            self.cluster_profiles_window = PlotWindow( 3, ( 1, 1, 1 ) )
        if create:
            self.cluster_profiles_window.show()

        if self.cluster_profiles_window != None:

            if self.pipeline.partition == None:
                self.cluster_profiles_window.hide()

            if self.cluster_profiles_window.isVisible():

                #self.draw_hcluster_dendrogram()

                self.draw_cluster_profile(
                    self.cluster_profiles_window,
                    0,
                    'Cluster profile of treatment %s' % self.pipeline.pdc.treatments[ treatmentId1 ].name,
                    treatmentId1,
                    facecolor='yellow'
                )

                self.draw_cluster_profile(
                    self.cluster_profiles_window,
                    1,
                    'Cluster profile of treatment %s' % self.pipeline.pdc.treatments[ treatmentId2 ].name,
                    treatmentId2,
                    facecolor='yellow'
                )

                self.draw_cluster_profile_comparison(
                    self.cluster_profiles_window,
                    2,
                    'Comparison of both cluster profiles',
                    treatmentId1,
                    treatmentId2
                )



    def create_cluster_profiles_window(self):

        try: self.cluster_profiles_window.close()
        except: pass


        print 'computing cluster profiles heatmap...'

        #profileHeatmap = numpy.zeros( ( self.pipeline.clusterProfiles.shape[0], self.pipeline.clusterProfiles.shape[0] ) )

        #normClusterProfiles = numpy.empty( self.pipeline.clusterProfiles.shape )

        """for i in xrange( self.pipeline.clusterProfiles.shape[0] ):

            profile1 = self.pipeline.clusterProfiles[ i ]
            #abs_value = numpy.sqrt( numpy.sum( profile1 ** 2 ) )
            abs_value = float( numpy.sum( profile1 ) )
            norm_profile1 = profile1 / abs_value
            normClusterProfiles[ i ] = norm_profile1

            for j in xrange( 0, i ):
    
                profile2 = self.pipeline.clusterProfiles[ j ]
                #abs_value = numpy.sqrt( numpy.sum( profile2 ** 2 ) )
                abs_value = float( numpy.sum( profile2 ) )
                norm_profile2 = profile2 / abs_value
                # L2-norm
                dist = numpy.sqrt( numpy.sum( ( norm_profile1 - norm_profile2 ) ** 2 ) )
                # chi-square
                #dist =  ( norm_profile1 - norm_profile2 ) ** 2 / ( norm_profile1 + norm_profile2 )
                #dist[ numpy.logical_and( norm_profile1 == 0, norm_profile2 == 0 ) ] = 0.0
                #dist = numpy.sum( dist )
                #print '%s <-> %s: %f' % ( self.pipeline.pdc.treatments[ i ].name, self.pipeline.pdc.treatments[ j ].name, dist )
    
                profileHeatmap[ i, j ] = dist
                profileHeatmap[ j, i ] = dist

        print 'done'"""

        #max = numpy.max( profileHeatmap )
        #profileHeatmap = profileHeatmap / max

        pageTitles = self.pipeline.clusterProfileLabelsDict.keys()

        self.cluster_profiles_window = MultiPageWindow(pageTitles,
                                                       title='Cluster profiles and distance heatmap')

        for k,group_description in enumerate(self.pipeline.clusterProfileLabelsDict):
            groups = grouping.get_groups(group_description, self.pdc,
                                         mask=self.pipeline.clusterCellMask,
                                         flat=False)
            clusterProfileLabels = self.pipeline.clusterProfileLabelsDict[group_description]
            clusterProfiles = self.pipeline.clusterProfilesDict[group_description]
            distanceHeatmap = self.pipeline.distanceHeatmapDict[group_description]

            window = ClusterProfilesWindow(
                clusterProfileLabels,
                clusterProfiles,
                distanceHeatmap,
                (1, 2)
            )
            self.cluster_profiles_window.set_child_window(k, window)

        self.cluster_profiles_window.page = 0
        self.cluster_profiles_window.show()

    def create_treatment_comparison_window(self):

        from ..core import debug
        debug.start_debugging()

        #try: self.contour_plot_window.close()
        #except: pass

        #pageList = [(2, 'Page 1'), (2, 'Page 2')]
        #self.contour_plot_window = MultiPagePlotWindow(pageList, show_toolbar=True, title='Contour')
        ##self.contour_plot_window = PlotWindow(number_of_plots=4, show_toolbar=True, title='Contour')

        #xlabel='Cells_Granularity_2_imGolgiMasked'
        #ylabel='Cells_Granularity_7_imGolgiMasked'
        #cellMask = self.pipeline.clusterCellMask
        #F = self.pipeline.clusterNormFeatures[:,:2]
        ##num_of_bins = 20
        ##edges = numpy.dstack(numpy.mgrid[-5:6:1,-5:6:1])
        #edges = [numpy.linspace(-5, 5, num=15), numpy.linspace(-5, 5, num=15)]
        #Z = numpy.histogramdd(F, edges)[0]
        #X = edges[0][:-1] + 0.5*(edges[0][1] - edges[0][0])
        #Y = edges[1][:-1] + 0.5*(edges[1][1] - edges[1][0])
        #XYZ = []
        #labels,masks = zip(*grouping.get_treatment_groups(self.pdc, cellMask))
        ##labels = []
        ##for tr in self.pdc.treatments:
        #for tr_mask in masks:
            ##tr_mask = self.pipeline.get_treatment_cell_mask(tr.index)[cellMask]
            #FF = F[tr_mask]
            #Z = numpy.histogramdd(FF, edges)[0]
            ##X1 = X[tr_mask]
            ##Y1 = Y[tr_mask]
            ##Z1 = Z[tr_mask]
            #XYZ.append([X,Y,Z])
            ##labels.append(tr.name)
        #colors = ['r','g','b']
        #marks = self.pipeline.clusters[:,:2]
        #self.contour_plot_window.page = 0
        #self.contour_plot_window.plot_window.draw_contour(
            #0, 'Cell densities', XYZ, labels, colors, marks=marks,
            #xlabel=xlabel, ylabel=ylabel,
            #title='Cell density contours'
        #)

        #from ..core import distance
        #tr_mask = self.pipeline.get_treatment_cell_mask(self.pdc.treatmentByName['wt'])[cellMask]
        #trans_m = distance.mahalanobis_transformation( F[tr_mask] )
        #F = distance.transform_features(F, trans_m)
        ##num_of_bins = 20
        #Z = numpy.histogramdd(F, edges)[0]
        #X = edges[0][:-1] + 0.5*(edges[0][1] - edges[0][0])
        #Y = edges[1][:-1] + 0.5*(edges[1][1] - edges[1][0])
        #XYZ = []
        #labels,masks = zip(*grouping.get_treatment_groups(self.pdc, cellMask))
        ##labels = []
        ##for tr in self.pdc.treatments:
        #for tr_mask in masks:
            ##tr_mask = self.pipeline.get_treatment_cell_mask(tr.index)[cellMask]
            #FF = F[tr_mask]
            #Z = numpy.histogramdd(FF, edges)[0]
            ##X1 = X[tr_mask]
            ##Y1 = Y[tr_mask]
            ##Z1 = Z[tr_mask]
            #XYZ.append([X,Y,Z])
            ##labels.append(tr.name)
        #colors = ['r','g','b']
        #marks = self.pipeline.clusters[:,:2]
        #mahalMarks = distance.transform_features(marks, trans_m)
        ##self.contour_plot_window.page = 0
        #self.contour_plot_window.plot_window.draw_contour(
            #1, 'Cell densities in wt mahalanobis space',
            #XYZ, labels, colors, marks=mahalMarks,
            #xlabel=xlabel, ylabel=ylabel,
            #title='Cell density contours'
        #)

        #from ..core import distance
        #tr_mask = self.pipeline.get_treatment_cell_mask(self.pdc.treatmentByName['noc'])[cellMask]
        #trans_m = distance.mahalanobis_transformation( F[tr_mask] )
        #F = distance.transform_features(F, trans_m)
        ##num_of_bins = 20
        #Z = numpy.histogramdd(F, edges)[0]
        #X = edges[0][:-1] + 0.5*(edges[0][1] - edges[0][0])
        #Y = edges[1][:-1] + 0.5*(edges[1][1] - edges[1][0])
        #XYZ = []
        #labels,masks = zip(*grouping.get_treatment_groups(self.pdc, cellMask))
        ##labels = []
        ##for tr in self.pdc.treatments:
        #for tr_mask in masks:
            ##tr_mask = self.pipeline.get_treatment_cell_mask(tr.index)[cellMask]
            #FF = F[tr_mask]
            #Z = numpy.histogramdd(FF, edges)[0]
            ##X1 = X[tr_mask]
            ##Y1 = Y[tr_mask]
            ##Z1 = Z[tr_mask]
            #XYZ.append([X,Y,Z])
            ##labels.append(tr.name)
        #colors = ['r','g','b']
        #mahalMarks = distance.transform_features(marks, trans_m)
        #self.contour_plot_window.page = 1
        #self.contour_plot_window.plot_window.draw_contour(
            #0, 'Cell densities in noc mahalanobis space',
            #XYZ, labels, colors, marks=mahalMarks,
            #xlabel=xlabel, ylabel=ylabel,
            #title='Cell density contours'
        #)

        #from ..core import distance
        #tr_mask = self.pipeline.get_treatment_cell_mask(self.pdc.treatmentByName['bfa'])[cellMask]
        #trans_m = distance.mahalanobis_transformation( F[tr_mask] )
        #F = distance.transform_features(F, trans_m)
        ##num_of_bins = 20
        #Z = numpy.histogramdd(F, edges)[0]
        #X = edges[0][:-1] + 0.5*(edges[0][1] - edges[0][0])
        #Y = edges[1][:-1] + 0.5*(edges[1][1] - edges[1][0])
        #XYZ = []
        #labels,masks = zip(*grouping.get_treatment_groups(self.pdc, cellMask))
        ##labels = []
        ##for tr in self.pdc.treatments:
        #for tr_mask in masks:
            ##tr_mask = self.pipeline.get_treatment_cell_mask(tr.index)[cellMask]
            #FF = F[tr_mask]
            #Z = numpy.histogramdd(FF, edges)[0]
            ##X1 = X[tr_mask]
            ##Y1 = Y[tr_mask]
            ##Z1 = Z[tr_mask]
            #XYZ.append([X,Y,Z])
            ##labels.append(tr.name)
        #colors = ['r','g','b']
        #mahalMarks = distance.transform_features(marks, trans_m)
        ##self.contour_plot_window.page = 1
        #self.contour_plot_window.plot_window.draw_contour(
            #1, 'Cell densities in bfa mahalanobis space',
            #XYZ, labels, colors, marks=mahalMarks,
            #xlabel=xlabel, ylabel=ylabel,
            #title='Cell density contours'
        #)

        #self.contour_plot_window.show()

        try:
            self.treatment_comparison_window.close()
        except: pass

        try:
            self.dendrogram_window.close()
        except: pass

        plots_per_page, ok = \
                      QInputDialog.getInt(self, 'Treatment comparison', 'Number of plots per page:',
                                          value=2, min=0,
                                          max=10)
        if not ok:
            plots_per_page = 2

        pageList = [(1, group_description) for group_description in self.pipeline.clusterProfileLabelsDict]
        pageNum, pageTitles = zip(*pageList)

        self.treatment_comparison_window = MultiPageWindow(pageTitles,
                                                           title='Treatment comparison')
        self.dendrogram_window = MultiPagePlotWindow(pageList, show_toolbar=True, title='Dendrogram')

        for k,group_description in enumerate(self.pipeline.clusterProfileLabelsDict):
            groups = grouping.get_groups(group_description, self.pdc,
                                         mask=self.pipeline.clusterCellMask,
                                         flat=False)
            clusterProfileLabels = self.pipeline.clusterProfileLabelsDict[group_description]
            distanceHeatmap = self.pipeline.distanceHeatmapDict[group_description]
            dendrogram = self.pipeline.dendrogramDict[group_description]

            mask1 = numpy.isfinite(distanceHeatmap)[0]
            mask2 = numpy.isfinite(distanceHeatmap)[:,0]
            valid_mask = numpy.logical_or(mask1, mask2)
            distanceHeatmap = distanceHeatmap[valid_mask][:,valid_mask]

            labels = list(itertools.compress(clusterProfileLabels, valid_mask))
            i = 0
            j = 0
            pageList = []
            neighbourList = []
            def appendToPageList(numOfPlots, pageName):
                pageList.append((numOfPlots, pageName))
            ids = []
            for subgroups in groups:
                neighbours = []
                if len(subgroups) >= 0 and type(subgroups[0]) not in (list, tuple):
                    if valid_mask[j]:
                        ids.append(i)
                        neighbours.append(i)
                        i += 1
                    j += 1
                    if len(ids) >= plots_per_page:
                        appendToPageList(len(ids), 'Page %d' % (len(pageList)+1))
                        ids = []
                else:
                    subgroups = grouping.flatten_groups(subgroups)
                    ids = []
                    for subgroup in subgroups:
                        if valid_mask[j]:
                            ids.append(i)
                            neighbours.append(i)
                            i += 1
                        j += 1
                        if len(ids) >= plots_per_page:
                            appendToPageList(len(ids), 'Page %d' % (len(pageList)+1))
                            ids = []
                    if len(ids) > 0:
                        appendToPageList(len(ids), 'Page %d' % (len(pageList)+1))
                        ids = []
                neighbourList.extend(len(neighbours) * [neighbours])
            if len(ids) > 0:
                appendToPageList(len(ids), 'Page %d' % (len(pageList)+1))
            #pageList = []
            #labels = list(itertools.compress(clusterProfileLabels, valid_mask))
            #plotsPerPage = 4
            #for i in xrange(len(labels)):
            #    if (i+1) % plotsPerPage == 0:
            #        pageList.append([plotsPerPage, 'Page %d' % (len(pageList) + 1)])
            #if (i+1) % plotsPerPage > 0:
            #    pageList.append([(i+1) % plotsPerPage, 'Page %d' % (len(pageList) + 1)])
            window = TreatmentComparisonWindow(
                pageList, labels, labels,
                distanceHeatmap, neighbourList,
                show_menu=True
            )
            self.treatment_comparison_window.set_child_window(k, window)

            custom_kwargs = {
                'Z' : dendrogram,
                'labels' : labels,
                'title' : 'Dendrogram',
                'xlabel' : 'Treatment',
                'ylabel' : 'Distance'
            }
            self.dendrogram_window.get_plot_window(k).draw_custom(
                0, 'Dendrogram', print_engine.draw_dendrogram,
                custom_kwargs=custom_kwargs, want_figure=False
            )
            self.dendrogram_window.show()

        self.treatment_comparison_window.page = 0
        self.dendrogram_window.page = 0

        #distanceHeatmap = self.pipeline.nonControlReplicateDistanceHeatmap
        #mask1 = numpy.isfinite(distanceHeatmap)[0]
        #mask2 = numpy.isfinite(distanceHeatmap)[:,0]
        #valid_mask = numpy.logical_or(mask1, mask2)
        #distanceHeatmap = distanceHeatmap[valid_mask][:,valid_mask]
        #pageList = []
        #plotLabels = []
        #labels = []
        #i = 0
        #j = 0
        #for tr in self.pipeline.pdc.treatments:
            #ids = []
            #for repl in self.pipeline.pdc.replicates:
                #if valid_mask[j]:
                    #ids.append(i)
                    #plotLabels.append('Treatment %s, Replicate %d' % (tr.name,repl.index))
                    #labels.append('%s %d' % (tr.name, repl.index))
                    #i += 1
                #j += 1
            #if len(ids) > 0:
                #pageList.append((len(ids), 'Treatment %s' % tr.name))
        #self.treatment_replicate_comparison_window = TreatmentComparisonWindow(
            #pageList, plotLabels, labels,
            #self.pipeline.pdc,
            #distanceHeatmap
        #)
        #self.treatment_replicate_comparison_window.show()

        #try: self.treatment_comparison_window.close()
        #except: pass

        #distanceHeatmap = self.pipeline.nonControlDistanceHeatmap
        #mask1 = numpy.isfinite(distanceHeatmap)[0]
        #mask2 = numpy.isfinite(distanceHeatmap)[:,0]
        #valid_mask = numpy.logical_or(mask1, mask2)
        #distanceHeatmap = distanceHeatmap[valid_mask][:,valid_mask]
        #pageList = []
        #plotLabels = []
        #labels = []
        #i = 0
        #for tr in self.pipeline.pdc.treatments:
            #ids = []
            #if valid_mask[i]:
                #ids.append(i)
                #plotLabels.append('Treatment %s' % tr.name)
                #labels.append('%s' % tr.name)
                #i += 1
            #if len(ids) > 0:
                #pageList.append((len(ids), 'Treatment %s' % tr.name))
        #self.treatment_comparison_window = TreatmentComparisonWindow(
            #pageList, plotLabels, labels,
            #self.pipeline.pdc,
            #distanceHeatmap
        #)
        self.treatment_comparison_window.show()

        debug.suspend_debugging()

    def on_draw_clustering_plots(self, create=True):

        #self.on_draw_cluster_profiles( 0, 1, True )

        self.create_cluster_profiles_window()

        self.create_treatment_comparison_window()

        #self.on_draw_hcluster_plots( create )

        #return

        try:
            self.clustering_plot_window
        except:
            self.clustering_plot_window = None
        if self.clustering_plot_window == None and create:
            self.clustering_plot_window = PlotWindow(1, ( 3, 1 ))
        if create:
            self.clustering_plot_window.show()

        if self.clustering_plot_window != None:

            if self.pipeline.partition == None:
                self.clustering_plot_window.hide()

            if self.clustering_plot_window.isVisible():

                #self.draw_hcluster_dendrogram()

                #self.draw_inter_cluster_distances(
                #    self.clustering_plot_window,
                #    0,
                #    'Pairwise distances between the clusters',
                #    self.pipeline.nonControlInterClusterDistances
                #)

                self.draw_cluster_population(
                    self.clustering_plot_window,
                    0,
                    'Population size of each cluster',
                    self.pipeline.clusters,
                    self.pipeline.partition
                )

                """self.draw_cluster_stddevs(
                    self.clustering_plot_window,
                    1,
                    'Stddev of each cluster',
                    self.pipeline.clusters,
                    self.pipeline.partition,
                    self.pipeline.nonControlNormFeatures
                )"""


    def draw_total_treatment_population(self, plot_window, plot_index, caption):

        # retrieve data for treatment population
        population = numpy.empty( len( self.pdc.treatments ) )
        x = numpy.arange( len( self.pdc.treatments ) )
        x_labels = []
        for tr in self.pdc.treatments:
            tr_mask = self.pdc.objFeatures[ : , self.pdc.objTreatmentFeatureId ] == tr.index
            tr_population = numpy.sum( tr_mask )
            population[ tr.index ] = tr_population
            x_labels.append( tr.name )
        plot_window.draw_barplot( plot_index, caption, population, x, x_labels, facecolor='blue' )

    def draw_nonControl_treatment_population(self, plot_window, plot_index, caption):

        # retrieve data for treatment population
        population = numpy.empty( len( self.pdc.treatments ) )
        x = numpy.arange( len( self.pdc.treatments ) )
        x_labels = []
        for tr in self.pdc.treatments:
            tr_mask = self.features[ : , self.pdc.objTreatmentFeatureId ] == tr.index
            tr_population = numpy.sum( tr_mask )
            population[ tr.index ] = tr_population
            x_labels.append( tr.name )
        plot_window.draw_barplot( plot_index, caption, population, x, x_labels, facecolor='blue' )

    def draw_treatment_penetrance(self, plot_window, plot_index, caption):

        # retrieve data for treatment population
        penetrance = numpy.empty( len( self.pdc.treatments ) )
        x = numpy.arange( len( self.pdc.treatments ) )
        x_labels = []
        for tr in self.pdc.treatments:
            total_tr_mask = self.pdc.objFeatures[ : , self.pdc.objTreatmentFeatureId ] == tr.index
            nonControl_tr_mask = self.features[ : , self.pdc.objTreatmentFeatureId ] == tr.index
            total_tr_population = numpy.sum( total_tr_mask )
            nonControl_tr_population = numpy.sum( nonControl_tr_mask )
            penetrance[ tr.index ] = 100.0 * nonControl_tr_population / float( total_tr_population )
            x_labels.append( tr.name )
        plot_window.draw_barplot( plot_index, caption, penetrance, x, x_labels, facecolor='blue' )

    def draw_mahal_dist_control_treatment(self, plot_window, plot_index, caption):

        # retrieve data
        control_mask = self.pipeline.mask_and( self.pipeline.get_valid_cell_mask(), self.pipeline.get_control_treatment_cell_mask() )
        mahal_dist = self.pdc.objFeatures[ control_mask ][ : , self.pdc.objFeatureIds[ 'Mahalanobis Distance' ] ]
        mask = mahal_dist < 500
        mahal_dist = mahal_dist[ mask ]
        plot_window.draw_histogram( plot_index, caption, mahal_dist, 200, facecolor='yellow' )

    def draw_mahal_dist_non_control_treatment(self, plot_window, plot_index, caption):

        # retrieve data
        control_mask = self.pipeline.mask_and( self.pipeline.get_valid_cell_mask(), self.pipeline.get_non_control_treatment_cell_mask() )
        mahal_dist = self.pdc.objFeatures[ control_mask ][ : , self.pdc.objFeatureIds[ 'Mahalanobis Distance' ] ]
        mask = mahal_dist < 500
        mahal_dist = mahal_dist[ mask ]
        plot_window.draw_histogram( plot_index, caption, mahal_dist, 200, facecolor='green' )

    def draw_mahal_dist_non_control(self, plot_window, plot_index, caption):

        # retrieve data
        non_control_mask = self.pipeline.mask_and( self.pipeline.get_valid_cell_mask(), self.pipeline.get_non_control_cell_mask() )
        mahal_dist = self.pdc.objFeatures[ non_control_mask ][ : , self.pdc.objFeatureIds[ 'Mahalanobis Distance' ] ]
        mask = mahal_dist < 500
        mahal_dist = mahal_dist[ mask ]
        plot_window.draw_histogram( plot_index, caption, mahal_dist, 200, facecolor='blue' )


    def on_draw_general_plots(self, create=True):

        try:
            self.general_plot_window
        except:
            self.general_plot_window = None
        if self.general_plot_window == None and create:
            self.general_plot_window = PlotWindow(3, ( 1, 1, 1 ))
        if create:
            self.general_plot_window.show()

        if self.general_plot_window != None:

            print 'drawing general plots'

            if self.general_plot_window.isVisible():

                """self.draw_total_treatment_population(
                    self.general_plot_window,
                    0,
                    'Total population size of each treatment'
                )

                self.draw_nonControl_treatment_population(
                    self.general_plot_window,
                    1,
                    'Non-control population size of each treatment'
                )

                self.draw_treatment_penetrance(
                    self.general_plot_window,
                    2,
                    'Penetrance of each treatment'
                )"""

                self.draw_mahal_dist_control_treatment(
                    self.general_plot_window,
                    0,
                    'Mahalanobis distance of negative control cells'
                )
                self.draw_mahal_dist_non_control_treatment(
                    self.general_plot_window,
                    1,
                    'Mahalanobis distance of non-control cells'
                )
                self.draw_mahal_dist_non_control(
                    self.general_plot_window,
                    2,
                    'Mahalanobis distance of phenotypic cells'
                )

    def draw_cluster_profile(self, plot_window, plot_index, caption, groupId, **kwargs):

        values = self.pipeline.clusterProfiles[ groupId ]

        """trMask = self.features[ :, self.pdc.objTreatmentFeatureId ] == groupId

        print 'trMask: %d' % ( numpy.sum( trMask ) )
        values = numpy.zeros( ( self.pipeline.clusters.shape[0], ) )
        for i in xrange( self.pipeline.clusters.shape[0] ):
            values[ i ] = numpy.sum( self.partition[ trMask ] == i )
            print '%d: %d' % ( i, values[ i ] )

        print numpy.sum( values )"""

        bin_labels = []
        bin_rescale = None

        for i in xrange( self.pipeline.clusters.shape[0] ):
            label = '%d' % i
            bin_labels.append( label )

        x = numpy.arange( self.pipeline.clusters.shape[0] )

        plot_window.draw_barplot(plot_index, caption, values, x, bin_labels, **kwargs )

    def draw_intra_cluster_distribution(self, plot_window, plot_index, caption, groupId):

        distances = self.pipeline.intraClusterDistances

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
            tr_mask = self.features[ : , self.pdc.objTreatmentFeatureId ] == tr.index
            tr_obj_count = numpy.sum( tr_mask )
            ratio = numpy.sum( values[:] == tr.index )
            ratio /= float( tr_obj_count )
            ratio *= 100.0
            label = '%s\n%0.1f%%' % ( tr.name, ratio )
            bin_labels.append( label )
        plot_window.draw_histogram( plot_index, caption, values, bins, bin_labels, bin_rescale )

    def draw_treatment_population(self, plot_window, plot_index, caption, groupId):

        # retrieve data for cluster population
        population = numpy.empty( len( self.pdc.treatments ) )
        x = numpy.arange( len( self.pdc.treatments ) )
        x_labels = []
        for tr in self.pdc.treatments:
            tr_mask = self.features[ : , self.pdc.objTreatmentFeatureId ] == tr.index
            tr_population = numpy.sum( tr_mask )
            population[ tr.index ] = tr_population
            x_labels.append( tr.name )
        plot_window.draw_barplot( plot_index, caption, population, x, x_labels, facecolor='blue' )

    def draw_cluster_similarities(self, plot_window, plot_index, caption, groupId):

        similarities = self.pipeline.interClusterDistances[ groupId ]

        x = numpy.arange( similarities.shape[0] )

        plot_window.draw_barplot( plot_index, caption, similarities, x, facecolor='orange' )

        """weights = None
        if self.pipeline.nonControlWeights != None:
            # retrieve data for cluster population
            #weights = self.pipeline.nonControlWeights[ groupId ]
            weights = self.pipeline.nonControlWeights
            if len( weights.shape ) > 1:
                weights = weights[ groupId ]

        clusters = self.pipeline.clusters
        partition = self.pipeline.partition

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
        values = numpy.zeros( ( self.pipeline.clusters.shape[0], ) )
        for k in xrange( values.shape[0] ):
            values[ k ] = numpy.sum( new_partition == k )
        x = numpy.arange( values.shape[0] )

        plot_window.draw_barplot( plot_index, caption, values, x, facecolor='orange' )"""


    def draw_weights(self, plot_window, plot_index, caption, groupId=-1):

        if self.pipeline.nonControlWeights != None:
            # retrieve data for cluster population
            #weights = self.pipeline.nonControlWeights[ groupId ]
            weights = self.pipeline.nonControlWeights
            if len( weights.shape ) > 1:
                if groupId < 0:
                    groupId = self.picked_group
                    if groupId < 0:
                        return
                weights = weights[ groupId ]
            x = numpy.arange( weights.shape[0] )
            plot_window.draw_barplot( plot_index, caption, weights, x, facecolor='purple' )

    def draw_feature_importance(self, plot_window, plot_index, caption, groupId):

        try:
            self.__norm_features
        except:
            self.__norm_features, valid_mask = self.pipeline.compute_normalized_features( self.featureset_index )
            self.__feature_importance = numpy.empty(
                ( self.get_number_of_groups(), self.__norm_features.shape[ 1 ] )
            )
            for i in xrange( self.get_number_of_groups() ):
                point_mask = self.get_id_mask_for_group( i )
                self.__feature_importance[ i ] = self.pipeline.compute_feature_importance(
                    self.__norm_features,
                    point_mask
                )

        importance = self.__feature_importance[ groupId ]
        x = numpy.arange( importance.shape[0] )
        plot_window.draw_barplot( plot_index, caption, importance, x, facecolor='purple' )


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
            #    #self.statusBar().showMessage( 'You have to select a group first!', 2000 )
            #    return

            if self.group_policy == self.GROUP_POLICY_TREATMENT:
                if self.pipeline.partition == None:
                    self.__plot_window.set_number_of_plots( 1 )
                else:
                    self.__plot_window.set_number_of_plots( 2, ( 1, 1 ) )

            else:
                self.__plot_window.set_number_of_plots( 1 )

            if self.group_policy == self.GROUP_POLICY_TREATMENT:

                self.draw_treatment_population(
                    self.__plot_window,
                    0,
                    'Population size of each treatment',
                    groupId
                )

                #self.draw_feature_importance(
                #    self.__plot_window,
                #    0,
                #    'Feature importance',
                #    groupId
                #)

                if self.pipeline.partition != None:

                    self.draw_cluster_profile(
                        self.__plot_window,
                        1,
                        'Cluster profile of the selected treatment',
                        groupId
                    )

            else:

                self.draw_cluster_to_treatment_distribution(
                    self.__plot_window,
                    0,
                    'Treatments',
                    groupId
                )

                #if self.pipeline.partition != None:
                #
                #    self.draw_cluster_similarities(
                #        self.__plot_window,
                #        1,
                #        'Distance to other clusters',
                #        groupId
                #    )

                    #self.draw_intra_cluster_distribution(
                    #    self.__plot_window,
                    #    2,
                    #    'Clusters',
                    #    groupId
                    #)

                    #self.draw_weights(
                    #    self.__plot_window,
                    #    2,
                    #    'Feature importance',
                    #    groupId
                    #)


    def on_draw_histogram(self, groupId=-1, create=True):

        try:
            self.histogram
        except:
            self.histogram = None
        if self.histogram == None and create:
            self.histogram = PlotWindow( 2, ( 1, 1 ) )
        if create:
            self.histogram.show()

        if groupId < 0:
            groupId = self.picked_group

        if self.histogram != None and self.histogram.isVisible():

            if groupId < 0:
                self.statusBar().showMessage( 'You have to select a group first!', 2000 )
            else:
                index = self.histogram_combo.currentIndex()
                name = str( self.histogram_combo.itemData( index ).toString() )
                values,groups = self.select_data_by_name( name )
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
                            tr_mask = self.features[ : , self.pdc.objTreatmentFeatureId ] == tr.index
                            tr_obj_count = numpy.sum( tr_mask )
                            #bin_rescale[ tr.index ] = 100.0 / tr_obj_count
                            ratio = numpy.sum( values[:] == tr.index )
                            ratio /= float( tr_obj_count )
                            ratio *= 100.0
                            if tr_obj_count == 0:
                                ratio = 0
                            #ratio = values[ tr.index ] / float( tr_obj_count ) * 100.0
                            #ratio = values[ tr.index ]
                            #ratio = tr_obj_count
                            label = '%s\n%d%%' % ( tr.name, int( ratio ) )
                            bin_labels.append( label )
                value_name = str( self.histogram_combo.currentText() )
                plot_label = 'Histogram of %s for %s %d' % ( value_name, self.get_group_policy_name(), self.picked_group )
                self.histogram.draw_histogram( 0, plot_label, values, bins, bin_labels, bin_rescale )
                self.histogram.show()

                #norm_values = self.pipeline.nonControlNormFeatures[]
                norm_values = values
                x, edf = self.compute_edf( norm_values )
                self.histogram.draw_lineplot( 1, 'EDF', x, edf, fillstyle='bottom' )

    def compute_edf(self, values):
        v = values.copy()
        v.sort()
        x = numpy.empty( ( 2 * v.shape[0], ) )
        edf = numpy.empty( ( 2 * v.shape[0], ) )
        for i in xrange( v.shape[0] ):
            x[ 2*i ] = v[ i ]
            x[ 2*i + 1 ] = v[ i ]
            edf[ 2*i ] = i
            edf[ 2*i + 1 ] = i + 1
        return x, edf

    def on_cluster_param1_changed(self, param1):
        self.cluster_param1 = param1

    def on_cluster_param2_changed(self, param2):
        self.cluster_param2 = param2

    def on_cluster_param3_changed(self, param3):
        self.cluster_param3 = param3

    def on_cluster_param4_changed(self, param4):
        self.cluster_param4 = param4

    def on_cluster_exp_factor_changed(self, exp_factor):
        self.cluster_exp_factor = exp_factor

    def on_cluster_combo_changed(self, index):
        self.featureset_index = index

    def on_featureset_changed(self, index):
        self.featureset_index = index

        try:
            self.__norm_features, valid_mask = self.pipeline.compute_normalized_features( self.featureset_index )
            self.__feature_importance = numpy.empty(
                ( self.get_number_of_groups(), self.__norm_features.shape[ 1 ] )
            )
            for i in xrange( self.get_number_of_groups() ):
                point_mask = self.get_id_mask_for_group( i )
                self.__feature_importance[ i ] = self.pipeline.compute_feature_importance(
                    self.__norm_features,
                    point_mask
                )
            self.on_draw_plots( -1, False )
        except:
            pass


    def on_pipeline_finished(self):


        import time
        t2 = time.time()
        c2 = time.clock()
        dt = t2 - self.__clustering_t1
        dc = c2 - self.__clustering_c1
        print 'dt: %.2f' % dt
        print 'dc: %.2f' % dc

        try:
            result = self.pipeline.get_result()
        except Exception, e:
            result = False

        if result:
    
            if self.__clustering_done:

                self.statusBar().showMessage( 'Finished clustering!', 2000 )

                self.__pipeline_running = False
                self.pipeline.disconnect( self.pipeline, pipeline.SIGNAL('finished()'), self.on_pipeline_finished )
                self.partition = self.pipeline.partition
                #self.features = self.pipeline.nonControlFeatures
                self.features = self.pipeline.clusterFeatures
                self.clusteringRadiobutton.setEnabled( True )
                self.clusteringRadiobutton.setChecked( True )

                self.group_combo.clear()
                for i in xrange( self.get_number_of_groups() ):
                    self.group_combo.addItem( str( i ), i )

                self.group_combo.setCurrentIndex( -1 )

                self.on_group_policy_changed( self.GROUP_POLICY_CLUSTERING )
            else:
                self.__clustering_done = True
                self.pipeline.start_cluster_profiling(
                    self.cluster_method, self.hcluster_method, self.cluster_exp_factor,
                    self.cluster_profile_metric, self.selected_groups)
        else:
            self.statusBar().showMessage( 'Error while running pipeline' )
            self.__pipeline_running = False
            self.pipeline.disconnect( self.pipeline, pipeline.SIGNAL('finished()'), self.on_pipeline_finished )

    def on_cluster_button(self):

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
            partition = self.pipeline.partition
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

        if not self.__pipeline_running:

            self.selected_groups = map(QListWidgetItem.text, self.groups_listwidget.selectedItems())
            self.selected_groups = map(str, self.selected_groups)
            if len(self.selected_groups) <= 0:
                QMessageBox(
                    QMessageBox.Warning,
                    'Warning',
                    'At least one group has to be selected',
                    QMessageBox.Ok,
                    self
                ).exec_()
                return

            self.statusBar().showMessage( 'Running clustering...' )
    
            self.cluster_method = str(self.cluster_method_combo.itemData(self.cluster_method_combo.currentIndex()).toString())
            self.hcluster_method = str(self.hcluster_method_combo.itemData(self.hcluster_method_combo.currentIndex()).toString())
            self.cluster_profile_metric = str(self.cluster_profile_metric_combo.itemData(self.cluster_profile_metric_combo.currentIndex()).toString())
            #self.pipeline.run_clustering( method, self.featureset_index, self.cluster_param1, self.cluster_param2, self.cluster_param3 )

            self.pipeline.connect(self.pipeline, pipeline.SIGNAL('finished()'), self.on_pipeline_finished)

            self.__pipeline_running = True
            self.__clustering_done = False
            import time
            self.__clustering_t1 = time.time()
            self.__clustering_c1 = time.clock()

            self.pipeline.start_clustering(
                self.selected_groups[0], self.cluster_method, self.featureset_index,
                self.cluster_param1, self.cluster_param2, self.cluster_param3,
                self.cluster_param4, self.cluster_exp_factor
            )

        """self.partition = self.pipeline.partition

        self.features = self.pipeline.nonControlFeatures
        #self.mahalFeatures = self.pipeline.nonControlTransformedFeatures
        #self.featureNames = self.pipeline.featureNames
        #self.partition = self.pipeline.partition
        #self.number_of_clusters = self.pipeline.clusters.shape[0]

        self.statusBar().showMessage( 'Finished clustering!', 2000 )

        #self.featureset_combo.setEnabled( True )
        #self.featureset_label.setEnabled( True )

        self.clusteringRadiobutton.setEnabled( True )
        self.clusteringRadiobutton.setChecked( True )

        self.group_combo.clear()
        for i in xrange( self.get_number_of_groups() ):
            self.group_combo.addItem( str( i ), i )

        self.group_combo.setCurrentIndex( -1 )

        self.on_group_policy_changed( self.GROUP_POLICY_CLUSTERING )"""

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

        return


    def on_group_policy_changed(self, group_policy):

        self.group_policy = group_policy

        if self.group_policy == self.GROUP_POLICY_CLUSTERING:
            self.features = self.pdc.objFeatures[self.pipeline.clusterCellMask]
            self.sorting_by_group = numpy.argsort( self.partition )
        elif self.group_policy == self.GROUP_POLICY_TREATMENT:
            self.features = self.pdc.objFeatures[self.pipeline.nonControlCellMask]
            self.sorting_by_group = numpy.argsort( self.features[ : , self.pdc.objTreatmentFeatureId ] )

        self.picked_group = self.get_group_id( self.picked_index )

        objectId = self.features[ self.picked_index, self.pdc.objObjectFeatureId ]
        self.mpl_toolbar.set_selection(
            objectId,
            self.get_group_policy_name(),
            self.get_group_name()
        )

        self.group_combo.clear()
        for i in xrange( self.get_number_of_groups() ):
            name = str( i )
            if self.group_policy == self.GROUP_POLICY_TREATMENT:
                name = self.pdc.treatments[ i ].name
            self.group_combo.addItem( name, i )

        #self.print_cluster_profiles_button.setEnabled( True )

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
            #self.create_cluster_profiles_window()
            #self.create_treatment_comparison_window()

        #for i in xrange( self.pipeline.clusters.shape[0] ):
        #    p_mask = self.pipeline.partition == i
        #    print 'cluster %d -> %d' % ( i,  numpy.sum( p_mask ) )

        self.on_draw_plots( -1, False )

        self.last_group = None
        self.last_index = None

        self.on_selection_changed( self.picked_group, self.picked_index, False )


    def on_view_cell_mask_changed(self, i):
        self.on_view_all_cells()

    def on_view_all_cells(self):

        cell_mask_name = str( self.__view_cell_mask_combo.currentText() )
        cell_mask, caption = self.__cell_mask_dict[ cell_mask_name ]

        try:
            cell_mask_filter_widget
        except:
            cell_mask_filter_widget = CellMaskFilterWidget( self.pipeline, self.gallery )

        self.view_cells( cell_mask, self.picked_index, caption, [ ( cell_mask_filter_widget, 'Filter' ) ] )



    def build_widget(self):

        self.main_frame = QWidget()

        # Create the mpl Figure and FigCanvas objects. 
        # 5x4 inches, 100 dots-per-inch
        #
        self.dpi = 100
        #self.fig = Figure((5.0, 4.0), dpi=self.dpi)
        self.fig = pyplot.figure()
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setParent(self.main_frame)

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
            name = str( i )
            if self.group_policy == self.GROUP_POLICY_TREATMENT:
                name = self.pdc.treatments[ i ].name
            self.group_combo.addItem( name, i )

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

        #self.print_cells_button = QPushButton( 'Print cell galleries' )
        #self.connect( self.print_cells_button, SIGNAL('clicked()'), self.on_print_cells )

        #self.print_cluster_profiles_button = QPushButton( 'Print cluster profiles' )
        #self.print_cluster_profiles_button.setEnabled( False )
        #self.connect( self.print_cluster_profiles_button, SIGNAL('clicked()'), self.on_print_cluster_profiles )

        hbox3 = QHBoxLayout()
        hbox3.addWidget( self.merge_cluster_button )
        hbox3.addWidget( self.merge_cluster_combo )

        vbox = QVBoxLayout()
        vbox.addLayout( hbox2 )
        vbox.addWidget( self.view_cells_button )
        vbox.addWidget( self.show_plots_button )
        vbox.addWidget( self.dissolve_cluster_button )
        vbox.addLayout( hbox3 )
        #vbox.addWidget( self.print_cells_button )
        #vbox.addWidget( self.print_cluster_profiles_button )

        hbox.addLayout( vbox )

        hbox2 = QHBoxLayout()
        hbox2.addWidget( treatmentRadiobutton )
        hbox2.addWidget( clusteringRadiobutton )
        groupBox = QGroupBox( 'Grouping policy' )
        groupBox.setLayout( hbox2 )

        hbox.addWidget( groupBox )


        #self.cluster_param1 = len( self.pdc.treatments )
        self.cluster_param1 = 250

        self.featureset_index = 0

        self.cluster_param1_spinbox = QSpinBox()
        self.cluster_param1_spinbox.setRange( 1, 100000 )
        self.cluster_param1_spinbox.setValue( self.cluster_param1 )
        self.connect( self.cluster_param1_spinbox, SIGNAL('valueChanged(int)'), self.on_cluster_param1_changed )

        cluster_button = QPushButton( 'Run clustering' )
        self.connect( cluster_button, SIGNAL('clicked()'), self.on_cluster_button )

        hbox2 = QHBoxLayout()
        hbox2.addWidget( QLabel( 'Param k:' ) )
        hbox2.addWidget( self.cluster_param1_spinbox, 1 )
        #hbox2.addWidget( cluster_button, 1 )

        self.cluster_param2 = -1
        self.cluster_param2_spinbox = QSpinBox()
        self.cluster_param2_spinbox.setRange( -1, 100000 )
        self.cluster_param2_spinbox.setValue( self.cluster_param2 )
        self.connect( self.cluster_param2_spinbox, SIGNAL('valueChanged(int)'), self.on_cluster_param2_changed )

        hbox4 = QHBoxLayout()
        hbox4.addWidget( QLabel( 'Param s:' ) )
        hbox4.addWidget( self.cluster_param2_spinbox, 1 )

        self.cluster_param3 = 2
        self.cluster_param3_spinbox = QSpinBox()
        self.cluster_param3_spinbox.setRange( 0, 100000 )
        self.cluster_param3_spinbox.setValue( self.cluster_param3 )
        self.connect( self.cluster_param3_spinbox, SIGNAL('valueChanged(int)'), self.on_cluster_param3_changed )

        hbox6 = QHBoxLayout()
        hbox6.addWidget( QLabel( 'Param p:' ) )
        hbox6.addWidget( self.cluster_param3_spinbox, 1 )

        self.cluster_param4 = 0
        self.cluster_param4_spinbox = QSpinBox()
        self.cluster_param4_spinbox.setRange( 0, 100000 )
        self.cluster_param4_spinbox.setValue( self.cluster_param4 )
        self.connect( self.cluster_param4_spinbox, SIGNAL('valueChanged(int)'), self.on_cluster_param4_changed )

        hbox7 = QHBoxLayout()
        hbox7.addWidget( QLabel( 'Param n:' ) )
        hbox7.addWidget( self.cluster_param4_spinbox, 1 )

        self.cluster_exp_factor = 20
        self.cluster_exp_factor_spinbox = QSpinBox()
        self.cluster_exp_factor_spinbox.setRange( -100, 100 )
        self.cluster_exp_factor_spinbox.setValue( self.cluster_exp_factor )
        self.connect( self.cluster_exp_factor_spinbox, SIGNAL('valueChanged(int)'), self.on_cluster_exp_factor_changed )

        hbox8 = QHBoxLayout()
        hbox8.addWidget( QLabel( 'Exp factor:' ) )
        hbox8.addWidget( self.cluster_exp_factor_spinbox, 1 )

        self.cluster_method_combo = QComboBox()
        for descr,name in cluster.get_hcluster_methods():
            self.cluster_method_combo.addItem( descr, name )
        self.cluster_method_combo.setCurrentIndex( 0 )

        hbox5 = QHBoxLayout()
        hbox5.addWidget( QLabel( 'Method:' ) )
        hbox5.addWidget( self.cluster_method_combo, 1 )

        self.hcluster_method_combo = QComboBox()
        k = 0
        for i,(descr,name) in enumerate(pipeline.Pipeline.get_hcluster_methods()):
            self.hcluster_method_combo.addItem(descr, name)
            if name == 'average':
                k = i
        self.hcluster_method_combo.setCurrentIndex(k)

        hbox10 = QHBoxLayout()
        hbox10.addWidget( QLabel( 'Hcluster:' ) )
        hbox10.addWidget( self.hcluster_method_combo, 1 )

        self.cluster_profile_metric_combo = QComboBox()
        for descr,name in [['Quadratic Chi','quadratic_chi'], ['L2 norm','l2_norm'],
                           ['Chi2 norm','chi2_norm'], ['Summed MinMax','summed_minmax']]:
            self.cluster_profile_metric_combo.addItem(descr, name)
        self.cluster_profile_metric_combo.setCurrentIndex( 0 )

        hbox9 = QHBoxLayout()
        hbox9.addWidget( QLabel( 'Profile metric:' ) )
        hbox9.addWidget( self.cluster_profile_metric_combo, 1 )

        self.featureset_label = QLabel( 'Feature set' )
        self.featureset_combo = QComboBox()
        for i in xrange( len( self.pipeline.clusterConfiguration ) ):
            name,config = self.pipeline.clusterConfiguration[ i ]
            self.featureset_combo.addItem( name, i )
        self.featureset_combo.setCurrentIndex( 0 )
        self.featureset_index = 0
        self.connect( self.featureset_combo, SIGNAL('currentIndexChanged(int)'), self.on_featureset_changed )
        hbox3 = QHBoxLayout()
        hbox3.addWidget( self.featureset_label )
        hbox3.addWidget( self.featureset_combo )

        self.groups_listwidget = QListWidget()
        self.groups_listwidget.setSelectionMode(QAbstractItemView.MultiSelection)
        group_descriptions = grouping.get_available_group_descriptions()
        for group_description in group_descriptions:
            self.groups_listwidget.addItem(group_description)
        self.groups_listwidget.setCurrentRow(0)

        vbox12 = QVBoxLayout()
        vbox12.addWidget( QLabel( 'Groups:' ) )
        vbox12.addWidget( self.groups_listwidget, 1 )

        vbox = QVBoxLayout()
        vbox.addLayout( hbox2 )
        vbox.addLayout( hbox4 )
        vbox.addLayout( hbox6 )  
        vbox.addLayout( hbox7 )
        vbox.addLayout( hbox8 )
        vbox.addLayout( hbox5 )
        vbox.addLayout( hbox10 )
        vbox.addLayout( hbox9 )
        if not self.__simple_ui:
            vbox.addLayout( hbox3 )
        vbox.addWidget( cluster_button )
        groupBox = QGroupBox( 'Clustering' )
        hbox11 = QHBoxLayout()
        hbox11.addLayout(vbox)
        hbox11.addLayout(vbox12)
        groupBox.setLayout(hbox11)
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

        #self.x_combo.addItem( 'mahalanobis distance', 'mahal_dist' )
        #self.y_combo.addItem( 'mahalanobis distance', 'mahal_dist' )

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

        #self.statusBar = QStatusBar()
        self.statusBarLabel = QLabel()
        self.statusBar().addWidget( self.statusBarLabel )

        self.__view_cell_mask_combo = QComboBox()
        default_index = 0
        i = 0
        for name in self.__cell_mask_dict:
            self.__view_cell_mask_combo.addItem( name )
            if name == self.__default_cell_mask_name:
                default_index = i
            i += 1
        self.__view_cell_mask_combo.setCurrentIndex( default_index )
        self.connect( self.__view_cell_mask_combo, SIGNAL('currentIndexChanged(int)'), self.on_view_cell_mask_changed )

        view_all_cells_button = QPushButton( 'View cells' )
        self.connect( view_all_cells_button, SIGNAL('clicked()'), self.on_view_all_cells )

        vbox2 = QVBoxLayout()
        vbox2.addWidget( self.__view_cell_mask_combo )
        vbox2.addWidget( view_all_cells_button )

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
        hbox.addLayout( vbox2 )
        #hbox.addWidget( view_all_cells_button )

        vbox = QVBoxLayout()
        vbox.addWidget(self.canvas, 1)
        vbox.addLayout( hbox )
        vbox.addLayout(vbox1)
        #vbox.addWidget(self.statusBar)

        self.main_frame.setLayout( vbox )

        self.setCentralWidget( self.main_frame )


    def on_load_pipeline(self):

        file_choices = "Pipeline state file (*.pis)"

        path = unicode(QFileDialog.getOpenFileName(self, 
                        'Open file', '', 
                        file_choices))
        if path:
            self.pipeline.load_state( str( path ) )

            self.partition = self.pipeline.partition
            self.features = self.pipeline.nonControlFeatures

            if self.partition != None:
                self.clusteringRadiobutton.setEnabled( True )
            else:
                self.clusteringRadiobutton.setEnabled( False )

            self.statusBar().showMessage( 'Loaded pipeline state from %s' % path )
            self.on_group_policy_changed( self.group_policy )

            return True

        return False

    def on_save_pipeline(self):

        file_choices = "Pipeline state file (*.pis)"

        path = unicode(QFileDialog.getSaveFileName(self, 
                        'Save file', '', 
                        file_choices))
        if path:
            self.pipeline.save_state( str( path ) )
            self.statusBar().showMessage( 'Saved pipeline state to %s' % path )
            return True

        return False

    def on_load_clusters(self):

        file_choices = "Pipeline cluster file (*.pic)"

        path = unicode(QFileDialog.getOpenFileName(self, 
                        'Open file', '', 
                        file_choices))
        if path:
            self.pipeline.load_clusters( str( path ), self.cluster_exp_factor )

            self.partition = self.pipeline.partition
            #self.features = self.pipeline.nonControlFeatures

            if self.partition != None:
                self.clusteringRadiobutton.setEnabled( True )
            else:
                self.clusteringRadiobutton.setEnabled( False )

            self.statusBar().showMessage( 'Loaded pipeline clusters from %s' % path )
            self.on_group_policy_changed( self.group_policy )

            return True

        return False

    def on_save_clusters(self):

        file_choices = "Pipeline cluster file (*.pic)"

        path = unicode(QFileDialog.getSaveFileName(self, 
                        'Save file', '', 
                        file_choices))
        if path:
            self.pipeline.save_clusters( str( path ) )
            self.statusBar().showMessage( 'Saved pipeline clusters to %s' % path )
            return True

        return False


    def build_menu(self):

        self.file_menu = self.menuBar().addMenu("&File")
        
        open_state_action = self.make_action("&Load pipeline state",
            shortcut="Ctrl+L", slot=self.on_load_pipeline, 
            tip="Load a pipeline state from a file")
        save_state_action = self.make_action("&Save pipeline state",
            shortcut="Ctrl+S", slot=self.on_save_pipeline, 
            tip="Save the pipeline state to a file")

        open_clusters_action = self.make_action("&Load pipeline clusters",
            shortcut="Ctrl+L", slot=self.on_load_clusters, 
            tip="Load a pipeline clusters from a file")
        save_clusters_action = self.make_action("&Save pipeline clusters",
            shortcut="Ctrl+S", slot=self.on_save_clusters, 
            tip="Save the pipeline clusters to a file")

        self.add_actions( self.file_menu, 
            ( open_state_action, save_state_action, None, open_clusters_action, save_clusters_action )
        )

        self.print_menu = self.menuBar().addMenu("&Print")
        population_and_penetrance_action = self.make_action("Print cell population and penetrance",
            shortcut="Ctrl+Q", slot=self.on_print_cell_population_and_penetrance, 
            tip="Print cell population and penetrance")
        selection_action = self.make_action("Print cell selection plots",
            shortcut="Ctrl+Q", slot=self.on_print_cell_selection_plots, 
            tip="Print cell selection plots")
        cluster_action = self.make_action("Print cluster &population",
            shortcut="Ctrl+P", slot=self.on_print_cluster_population, 
            tip="Print cluster population")
        profile_action = self.make_action("Print &cluster profiles",
            shortcut="Ctrl+C", slot=self.on_print_cluster_profiles, 
            tip="Print cluster profiles")
        norm0_profile_action = self.make_action("Print &normalized 0 cluster profiles",
            shortcut="Ctrl+N", slot=self.on_print_norm0_cluster_profiles, 
            tip="Print normalized 0 cluster profiles")
        norm1_profile_action = self.make_action("Print &normalized 1 cluster profiles",
            shortcut="Ctrl+N", slot=self.on_print_norm1_cluster_profiles, 
            tip="Print normalized 1 cluster profiles")
        norm2_profile_action = self.make_action("Print &normalized 2 cluster profiles",
            shortcut="Ctrl+N", slot=self.on_print_norm2_cluster_profiles, 
            tip="Print normalized 2 cluster profiles")
        norm3_profile_action = self.make_action("Print &normalized 3 cluster profiles",
            shortcut="Ctrl+N", slot=self.on_print_norm3_cluster_profiles, 
            tip="Print normalized 3 cluster profiles")
        norm4_profile_action = self.make_action("Print &normalized 4 cluster profiles",
            shortcut="Ctrl+N", slot=self.on_print_norm4_cluster_profiles, 
            tip="Print normalized 4 cluster profiles")
        all_profile_action = self.make_action("Print &all cluster profiles",
            shortcut="Ctrl+A", slot=self.on_print_all_cluster_profiles, 
            tip="Print all cluster profiles")
        gallery_action = self.make_action("Print cell &galleries",
            shortcut="Ctrl+G", slot=self.on_print_cells, 
            tip="Print cell galleries")

        self.add_actions( self.print_menu, 
            ( population_and_penetrance_action, selection_action, None,
              cluster_action, profile_action, norm0_profile_action, norm1_profile_action, norm2_profile_action, norm3_profile_action, norm4_profile_action, all_profile_action, gallery_action )
        )


    def add_actions(self, target, actions):
        for action in actions:
            if action is None:
                target.addSeparator()
            else:
                target.addAction(action)

    def make_action(  self, text, slot=None, shortcut=None, 
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
