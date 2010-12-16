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
from histogram_window import HistogramWindow
from gui_utils import CellPixmapFactory,CellFeatureTextFactory



TMP_IMAGE_FILENAME_TEMPLATE = '/dev/shm/adc-tmp-image-file-%d.tiff'



class ResultsWindow(QWidget):

    GROUP_POLICY_TREATMENT = 0
    GROUP_POLICY_CLUSTERING = 1
    DEFAULT_GROUP_POLICY = GROUP_POLICY_TREATMENT

    def __init__(self, pipeline, channelMapping, channelDescription, parent=None):

        self.pipeline = pipeline

        self.adc = pipeline.adc

        self.features = pipeline.nonControlFeatures
        self.mahalFeatures = pipeline.nonControlTransformedFeatures
        self.featureNames = pipeline.featureNames
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
            self.sorting_by_group = numpy.argsort( self.features[ : , self.adc.objTreatmentFeatureId ] )

        QWidget.__init__(self, parent)
        self.setWindowTitle('APC Results')

        self.build_widget()

        featureDescription = dict( self.adc.objFeatureIds )

        self.mahalFeatureIdOffset = max( featureDescription.values() ) + 1
        for i in xrange( self.mahalFeatures.shape[ 1 ] ):
            featureDescription[ 'mahal_' + self.featureNames[ i ] ] = i + self.mahalFeatureIdOffset

        self.gallery = GalleryWindow( featureDescription, self.channelMapping, self.channelDescription )

        self.histogram = None
        self.barplot = None

        self.on_draw()


    def closeEvent(self, ce):
        self.gallery.close()
        if self.histogram:
            self.histogram.close()
        if self.barplot:
            self.barplot.close()
        ce.accept()

    """def save_plot(self):
        file_choices = "PNG (*.png)|*.png"

        path = unicode(QFileDialog.getSaveFileName(self, 
                        'Save file', '', 
                        file_choices))
        if path:
            self.canvas.print_figure(path, dpi=self.dpi)
            self.statusBar().showMessage('Saved to %s' % path, 2000)"""


    def update_selection(self, groupId, focusId=-1):

        if self.histogram != None and self.histogram.isVisible():
            self.on_draw_histogram()

        selectionMask = self.get_id_mask_for_group( groupId )
    
        selectionIds = self.adc.objFeatures[ : , self.adc.objObjectFeatureId ][ selectionMask ]
    
        featureFactory = CellFeatureTextFactory( self.adc, self.pipeline.nonControlCellMask, self.mahalFeatures )
        pixmapFactory = CellPixmapFactory( self.adc, self.channelMapping, self.pipeline.nonControlCellMask )

        self.gallery.on_selection_changed(focusId, selectionIds, pixmapFactory, featureFactory)
        self.gallery.show()


    def on_selection_changed(self, groupId, focusId=-1):

        self.group_combo.setCurrentIndex( groupId )

        self.picked_index = focusId
        self.picked_group = groupId
        if groupId < 0:
            self.histogram_button.setEnabled( False )
            self.gallery.close()
            self.histogram.close()
        else:
            self.histogram_button.setEnabled( True )
            self.update_selection( groupId, focusId )


    def on_group_combo_activated(self, groupId):
        self.on_selection_changed( groupId )


    def on_pick(self, event):
        # The event received here is of the type
        # matplotlib.backend_bases.PickEvent
        #
        # It carries lots of information, of which we're using
        # only a small amount here.

        if event.mouseevent.button == 1:
            lines = event.artist
            ind = event.ind

            mx = event.mouseevent.xdata
            my = event.mouseevent.ydata
            x_lim = self.axes.get_xlim()
            y_lim = self.axes.get_ylim()
            x_weight = 1 / ( x_lim[1] - x_lim[0] )
            y_weight = 1 / ( y_lim[1] - y_lim[0] )
            d = ( x_weight * ( self.x_data - mx ) ) ** 2 + ( y_weight * ( self.y_data - my ) ) ** 2
            min_i = 0
            min_d = d[min_i]
            for i in xrange(d.shape[0]):
                if d[i] < min_d:
                    min_i = i
                    min_d = d[i]
            index = min_i

            if index < 0:
                self.statusBar.showMessage('you picked an unknown item', 2000)
                return

            self.statusBar.showMessage('you picked item %d' % index, 2000)

            focusId = index
            groupId = self.get_group_id( index )

            self.picked_group = groupId
            self.picked_index = focusId

            self.on_selection_changed( groupId, focusId )


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
                        markers[ min( i, len(markers)-1 ) ],
                        picker = 4,
                        label = '%d' % i
                )
            #if self.picked_index < 0:
            #    self.axes.plot(self.x_data[self.picked_index],
            #                   self.y_data[self.picked_index],
            #                   picked_symbol, label = 'selected')

        self.canvas.draw()



    def get_group_id(self, id):

        if self.group_policy == self.GROUP_POLICY_CLUSTERING:
            return int( self.partition[ id ] )
        elif self.group_policy == self.GROUP_POLICY_TREATMENT:
            return int( self.features[ id , self.adc.objTreatmentFeatureId ] )


    def get_id_mask_for_group(self, groupId):

        if self.group_policy == self.GROUP_POLICY_CLUSTERING:
            return self.partition == groupId
        elif self.group_policy == self.GROUP_POLICY_TREATMENT:
            tr = self.adc.treatments[ groupId ]
            return self.features[ : , self.adc.objTreatmentFeatureId ] == tr.rowId


    def get_number_of_groups(self):

        if self.group_policy == self.GROUP_POLICY_CLUSTERING:
            return self.number_of_clusters
        elif self.group_policy == self.GROUP_POLICY_TREATMENT:
            return len( self.adc.treatments )


    def group_data_by_group(self, values):

        groups = []

        for i in xrange( self.get_number_of_groups() ):

            selection_mask = self.get_id_mask_for_group( i )
            selection = values[ selection_mask ]
            groups.append( selection )

        return groups


    def select_data(self, data_name):

        data_name = str( data_name )

        if data_name.startswith( 'feature ' ):
            feature_id = int( data_name[ len( 'feature ' ) : ] )
            values = self.features[ : , feature_id ]

        elif data_name.startswith( 'mahal ' ):
            feature_id = int( data_name[ len( 'mahal ' ) : ] )
            values = self.mahalFeatures[ : , feature_id ]

        elif data_name == 'n':
            values = numpy.empty( self.features.shape[0] )
            for i in xrange( values.shape[0] ):
                values[i] = i

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
        self.x_data,self.x_data_by_group = self.select_data(name)

    def load_y_data(self, index):
        name = self.y_combo.itemData( index ).toString()
        self.y_data,self.y_data_by_group = self.select_data(name)

    def on_x_combo_changed(self, index):
        self.load_x_data( index )
        self.on_draw()

    def on_y_combo_changed(self, index):
        self.load_y_data( index )
        self.on_draw()

    def on_histogram_changed(self, i):
        self.on_draw_histogram()

    def on_draw_histogram(self):
        if self.picked_group < 0:
            self.statusBar.showMessage( 'You have to select a group first!', 2000 )
        else:
            index = self.histogram_combo.currentIndex()
            name = str( self.histogram_combo.itemData( index ).toString() )
            values,groups = self.select_data( name )
            if name.startswith( 'feature ' ):
                feature_id = int( name[ len( 'feature ' ) : ] )
            else:
                feature_id = -1
            groupId = self.picked_group
            values = groups[ groupId ]
            bins = self.histogram_spinbox.value()
            if self.histogram == None:
                self.histogram = HistogramWindow()
            bin_labels = None
            if feature_id == self.adc.objTreatmentFeatureId:
                if bins == len( self.adc.treatments ):
                    bin_labels = []
                    for tr in self.adc.treatments:
                        bin_labels.append( tr.name )
            self.histogram.draw_histogram( values, bins, bin_labels )
            self.histogram.show()


    def on_number_of_clusters_changed(self, number_of_clusters):
        self.number_of_clusters = number_of_clusters

    def on_cluster_combo_changed(self, index):
        self.supercluster_index = index

    def on_supercluster_changed(self, index):
        self.supercluster_index = index

    def on_cluster_button(self):

        self.statusBar.showMessage( 'Running clustering...' )

        self.pipeline.run_pipeline( self.supercluster_index )
        self.pipeline.run_clustering( self.number_of_clusters )

        self.features = self.pipeline.nonControlFeatures
        self.mahalFeatures = self.pipeline.nonControlTransformedFeatures
        self.featureNames = self.pipeline.featureNames
        self.partition = self.pipeline.nonControlPartition
        self.number_of_clusters = self.pipeline.nonControlClusters.shape[0]

        self.statusBar.showMessage( 'Finished clustering!', 2000 )

        self.supercluster_combo.setEnabled( True )
        self.supercluster_label.setEnabled( True )

        self.clusteringRadiobutton.setEnabled( True )
        self.clusteringRadiobutton.setChecked( True )

        self.group_combo.clear()
        for i in xrange( self.get_number_of_groups() ):
            self.group_combo.addItem( str( i ), i )

        self.group_combo.setCurrentIndex( -1 )

        if self.group_policy == self.GROUP_POLICY_CLUSTERING:
            self.on_group_policy_changed( self.GROUP_POLICY_CLUSTERING )

        self.barplot = HistogramWindow()
        silhouette = self.pipeline.nonControlSilhouette
        print silhouette
        print silhouette.shape
        self.barplot.draw_barplot( silhouette )
        self.barplot.show()


    def on_group_policy_changed(self, group_policy):
        self.group_policy = group_policy

        if self.group_policy == self.GROUP_POLICY_CLUSTERING:
            self.sorting_by_group = numpy.argsort( self.partition )
        elif self.group_policy == self.GROUP_POLICY_TREATMENT:
            self.sorting_by_group = numpy.argsort( self.features[ : , self.adc.objTreatmentFeatureId ] )

        self.group_combo.clear()
        for i in xrange( self.get_number_of_groups() ):
            name = str( i )
            if self.group_policy == self.GROUP_POLICY_TREATMENT:
                name = self.adc.treatments[ i ].name
            self.group_combo.addItem( name, i )

        self.load_x_data( self.x_combo.currentIndex() )
        self.load_y_data( self.y_combo.currentIndex() )
        self.on_draw()


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
        self.canvas.mpl_connect('pick_event', self.on_pick)

        # Create the navigation toolbar, tied to the canvas
        #
        self.mpl_toolbar = NavigationToolbar(self.canvas, self)

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

        hbox = QHBoxLayout()
        hbox.addWidget( treatmentRadiobutton )
        hbox.addWidget( clusteringRadiobutton )
        groupBox1 = QGroupBox( 'Grouping policy' )
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

        for i in xrange(self.mahalFeatures.shape[1]):
            self.x_combo.addItem('mahalanobis_%s' % self.featureNames[i],'mahal %d' % i)
            self.y_combo.addItem('mahalanobis_%s' % self.featureNames[i],'mahal %d' % i)

        keys = self.adc.objFeatureIds.keys()
        keys.sort()
        for name in keys:
            featureId = self.adc.objFeatureIds[ name ]
            self.x_combo.addItem(name,'feature %d' % featureId)
            self.y_combo.addItem(name,'feature %d' % featureId)

        self.connect(self.x_combo, SIGNAL('currentIndexChanged(int)'), self.on_x_combo_changed)
        self.connect(self.y_combo, SIGNAL('currentIndexChanged(int)'), self.on_y_combo_changed)
        self.x_combo.setCurrentIndex(1)
        self.y_combo.setCurrentIndex(2)


        hbox = QHBoxLayout()

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

        for i in xrange(self.mahalFeatures.shape[1]):
            self.histogram_combo.addItem('mahalanobis_%s' % self.featureNames[i],'mahal %d' % i)

        keys = self.adc.objFeatureIds.keys()
        keys.sort()
        for name in keys:
            featureId = self.adc.objFeatureIds[ name ]
            self.histogram_combo.addItem(name,'feature %d' % featureId)
            if featureId == self.adc.objTreatmentFeatureId:
                self.histogram_combo.setCurrentIndex( self.histogram_combo.count() - 1 )

        self.connect( self.histogram_combo, SIGNAL('currentIndexChanged(int)'), self.on_histogram_changed )

        self.histogram_button = QPushButton( 'Draw' )
        self.histogram_button.setEnabled( False )
        self.connect( self.histogram_button, SIGNAL('clicked()'), self.on_draw_histogram )

        self.histogram_spinbox = QSpinBox()
        self.histogram_spinbox.setRange( 1, 100 )
        self.histogram_spinbox.setValue( len( self.adc.treatments ) )
        self.connect( self.histogram_spinbox, SIGNAL('valueChanged(int)'), self.on_histogram_changed )

        hbox = QHBoxLayout()
        hbox.addWidget( self.histogram_combo, 1 )
        hbox.addWidget( QLabel( 'Bins:' ) )
        hbox.addWidget( self.histogram_spinbox )
        hbox.addWidget( self.histogram_button )
        groupBox3 = QGroupBox( 'Histogram' )
        groupBox3.setLayout( hbox )


        self.cluster_spinbox = QSpinBox()
        self.cluster_spinbox.setRange( 1, 100 )
        self.cluster_spinbox.setValue( len( self.adc.treatments ) )
        self.connect( self.cluster_spinbox, SIGNAL('valueChanged(int)'), self.on_number_of_clusters_changed )

        self.cluster_combo = QComboBox()
        for id in xrange( len( self.pipeline.clusterConfiguration ) ):
            name,featureNames = self.pipeline.clusterConfiguration[ id ]
            self.cluster_combo.addItem( name, id )
        self.connect( self.cluster_combo, SIGNAL('currentIndexChanged(int)'), self.on_cluster_combo_changed )
        self.cluster_combo.setCurrentIndex( -1 )
        self.cluster_combo.setCurrentIndex( 0 )

        cluster_button = QPushButton( 'Run clustering' )
        self.connect( cluster_button, SIGNAL('clicked()'), self.on_cluster_button )

        hbox = QHBoxLayout()
        hbox.addWidget( QLabel( 'Number of clusters:' ) )
        hbox.addWidget( self.cluster_spinbox, 1 )
        hbox.addWidget( QLabel( 'SuperCluster:' ) )
        hbox.addWidget( self.cluster_combo, 2 )
        hbox.addWidget( cluster_button, 1 )
        groupBox4 = QGroupBox( 'Clustering' )
        groupBox4.setLayout( hbox )


        self.group_combo = QComboBox()

        for i in xrange( self.get_number_of_groups() ):
            self.group_combo.addItem( str( i ), i )

        self.group_combo.setCurrentIndex( -1 )

        self.connect( self.group_combo, SIGNAL('activated(int)'), self.on_group_combo_activated )

        self.supercluster_label = QLabel( 'Select SuperCluster' )

        self.supercluster_combo = QComboBox()

        for i in xrange( len( self.pipeline.clusterConfiguration ) ):
            name,config = self.pipeline.clusterConfiguration[ i ]
            self.supercluster_combo.addItem( name, i )

        self.supercluster_combo.setCurrentIndex( 0 )
        self.supercluster_index = 0

        self.connect( self.supercluster_combo, SIGNAL('currentIndexChanged(int)'), self.on_supercluster_changed )
        self.supercluster_combo.setEnabled( False )
        self.supercluster_label.setEnabled( False )

        hbox = QHBoxLayout()
        hbox.addWidget( QLabel( 'Select group:' ) )
        hbox.addWidget( self.group_combo, 1 )
        hbox.addWidget( self.supercluster_label )
        hbox.addWidget( self.supercluster_combo, 1 )
        groupBox5 = QGroupBox( 'Grouping' )
        groupBox5.setLayout( hbox )


        #
        # Layout with box sizers
        #


        vbox1 = QVBoxLayout()

        vbox1.addWidget( groupBox1 )
        vbox1.addWidget( groupBox2 )
        vbox1.addWidget( groupBox3 )
        vbox1.addWidget( groupBox4 )
        vbox1.addWidget( groupBox5 )

        hbox = QHBoxLayout()

        hbox.addLayout(vbox1)

        self.statusBar = QStatusBar()

        vbox = QVBoxLayout()
        vbox.addWidget(self.canvas, 1)
        vbox.addWidget(self.mpl_toolbar)
        vbox.addLayout(hbox)
        vbox.addWidget(self.statusBar)

        self.setLayout(vbox)
