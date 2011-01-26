from PyQt4.QtCore import *
from PyQt4.QtGui import *


from ..core import parameter_utils as utils

from parameter_widgets import ParameterWidgetObjFeatures




class ClusterConfigurationWidget(QWidget):

    __pyqtSignals__ = ('configurationChanged',)

    def __init__(self, id, name, pdc):

        QWidget.__init__( self )

        self.id = id
        self.name = name

        self.featureIds = list( pdc.objFeatureIds.keys() )
        self.featureIds.sort()

        self.build_widget( name )

    def build_widget(self, name):

        label = QLabel( 'SuperCluster name:' )
        self.lineedit = QLineEdit()
        self.lineedit.setText( name )
        self.connect( self.lineedit, SIGNAL('textChanged(QString)'), self.on_change_name )

        hbox1 = QHBoxLayout()
        hbox1.addStretch( 1 )
        hbox1.addWidget( label, 0, Qt.AlignVCenter )
        hbox1.addWidget( self.lineedit, 0, Qt.AlignVCenter )
        hbox1.addStretch( 1 )

        self.listwidget = QListWidget()
        self.listwidget.setSelectionMode( QAbstractItemView.MultiSelection )

        for i in xrange( len( self.featureIds ) ):
            k = self.featureIds[ i ]
            self.listwidget.addItem( k )

        self.connect( self.listwidget, SIGNAL('itemSelectionChanged()'), self.on_selection_changed )

        vbox = QVBoxLayout()
        vbox.addWidget( self.listwidget, 1 )
        vbox.addLayout( hbox1 )
        self.setLayout( vbox )

    def update_configuration(self, config):
        for i in xrange( self.listwidget.count() ):
            item = self.listwidget.item( i )
            item.setSelected( str( item.text() ) in config )
        self.on_configuration_changed()

    def on_change_name(self, name):
        self.name = str( name )
        self.on_configuration_changed()

    def on_selection_changed(self):
        self.on_configuration_changed()

    def on_configuration_changed(self):
        value = []
        for item in self.listwidget.selectedItems():
            value.append( str( item.text() ) )

        self.emit( SIGNAL('configurationChanged'), self.id, self.name, value )



class ClusterConfigurationTab(QWidget):

    __OBJECT_NAME = 'ClusterConfiguration'

    def __init__(self, pdc, parent=None):

        QWidget.__init__( self, parent )

        self.pdc = pdc

        self.names = []
        for i in xrange( len( pdc.images[0].imageFiles ) ):
            name,path = pdc.images[0].imageFiles[i]
            self.names.append( name )

        #self.clusterConfiguration = {}
        self.clusterConfiguration = []

        self.superclusters = []

        self.build_widget()

        utils.register_object( self.__OBJECT_NAME )
        utils.register_attribute( self.__OBJECT_NAME, 'clusterConfiguration', self.getClusterConfiguration, self.setClusterConfiguration )

    def getClusterConfiguration(self):
        return self.clusterConfiguration
    def setClusterConfiguration(self, configurations):
        while len( self.superclusters ) > 0:
            self.on_remove_supercluster()
        #for k,v in value.iteritems():
        #    id = len( self.superclusters )
        #    self.add_supercluster( id, k )
        #    self.superclusters[ id ].update_configuration( value[ k ] )
        for id in xrange( len( configurations ) ):
            name,config = configurations[ id ]
            self.add_supercluster( id, name )
            self.superclusters[ id ].update_configuration( config )

    def on_change_configuration(self, id, name, config):
        #self.clusterConfiguration[ name ] = config
        self.supercluster_tab.setTabText( id, name )
        self.clusterConfiguration[ id ] = ( name, config )

    def add_supercluster(self, id, name):

        self.remove_supercluster_button.setEnabled( True )

        supercluster_widget = ClusterConfigurationWidget( id, name, self.pdc )
        self.superclusters.append( supercluster_widget )
        self.connect( supercluster_widget, SIGNAL('configurationChanged'), self.on_change_configuration )

        #self.clusterConfiguration[ name ] = []
        self.clusterConfiguration.append( ( name, [] ) )

        #index = self.vbox.count()
        index = self.supercluster_tab.addTab( supercluster_widget, name )
        self.supercluster_tab.setCurrentIndex( index )

    def on_add_supercluster(self):

        id = len( self.superclusters )
        name = 'SuperCluster %d' % ( id + 1 )

        self.add_supercluster( id, name )

    def on_remove_supercluster(self):
        index = self.supercluster_tab.currentIndex()
        supercluster_widget = self.superclusters[ index ]
        for sw in self.superclusters:
            if sw.id > index:
                sw.id -= 1
        #del self.clusterConfiguration[ supercluster_widget.name ]
        del self.clusterConfiguration[ index ]
        self.supercluster_tab.removeTab( self.supercluster_tab.indexOf( supercluster_widget ) )
        supercluster_widget.close()
        del self.superclusters[ index ]
        if len( self.superclusters ) <= 1:
            self.remove_supercluster_button.setEnabled( False )

    def build_widget(self):

        self.supercluster_tab = QTabWidget()

        add_supercluster_button = QPushButton( 'Add SuperCluster' )
        self.connect( add_supercluster_button, SIGNAL('clicked()'), self.on_add_supercluster )
        self.remove_supercluster_button = QPushButton( 'Remove SuperCluster' )
        self.connect( self.remove_supercluster_button, SIGNAL('clicked()'), self.on_remove_supercluster )
        self.remove_supercluster_button.setEnabled( False )

        hbox = QHBoxLayout()
        hbox.addWidget( add_supercluster_button )
        hbox.addWidget( self.remove_supercluster_button )
        
        vbox = QVBoxLayout()
        vbox.addWidget( self.supercluster_tab )
        vbox.addLayout( hbox )

        self.setLayout( vbox )

        self.on_add_supercluster()

