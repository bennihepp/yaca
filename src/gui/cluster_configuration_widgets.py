from PyQt4.QtCore import *
from PyQt4.QtGui import *


from ..core import parameter_utils as utils

from parameter_widgets import ParameterWidgetObjFeatures




class ClusterConfigurationWidget(QWidget):

    __pyqtSignals__ = ('configurationChanged',)

    def __init__(self, id, name, adc):

        QWidget.__init__( self )

        self.id = id
        self.name = name

        self.featureIds = list( adc.objFeatureIds.keys() )
        self.featureIds.sort()

        self.build_widget( name )

    def build_widget(self, name):

        label = QLabel( 'SuperCluster:' )
        self.lineedit = QLineEdit()
        self.lineedit.setText( name )
        self.connect( self.lineedit, SIGNAL('textChanged(QString)'), self.on_change_name )

        vbox1 = QVBoxLayout()
        vbox1.addStretch( 1 )
        vbox1.addWidget( label, 0, Qt.AlignVCenter )
        vbox1.addWidget( self.lineedit, 0, Qt.AlignVCenter )
        vbox1.addStretch( 1 )

        scrollarea = QScrollArea()
        self.buttongroup = QButtonGroup()
        self.buttongroup.setExclusive( False )
        self.buttons = []
        vbox2 = QVBoxLayout()

        for i in xrange( len( self.featureIds ) ):
            k = self.featureIds[ i ]
            self.checkBox = QCheckBox( k )
            self.checkBox.setChecked( False )
            vbox2.addWidget( self.checkBox )
            self.buttongroup.addButton( self.checkBox, i )
            self.buttons.append( self.checkBox )

        w = QWidget()
        w.setLayout( vbox2 )
        scrollarea.setWidget( w )

        self.connect( self.buttongroup, SIGNAL('buttonClicked(int)'), self.on_button_clicked )

        hbox = QHBoxLayout()
        hbox.addLayout( vbox1, 1 )
        hbox.addWidget( scrollarea, 2 )
        self.setLayout( hbox )

    def update_configuration(self, config):
        for i in xrange( len( self.buttons ) ):
            if str( self.buttons[ i ].text() ) in config:
                self.buttons[ i ].setChecked( True )
            else:
                self.buttons[ i ].setChecked( False )
        self.on_configuration_changed()

    def on_change_name(self, name):
        self.name = name
        self.on_configuration_changed()

    def on_button_clicked(self, value):
        self.on_configuration_changed()

    def on_configuration_changed(self):
        value = []
        for i in xrange( len( self.buttons ) ):
            if self.buttons[ i ].isChecked():
                value.append( self.featureIds[ i ] )

        self.emit( SIGNAL('configurationChanged'), self.id, self.name, value )



class ClusterConfigurationTab(QWidget):

    __OBJECT_NAME = 'ClusterConfiguration'

    def __init__(self, adc, parent=None):

        QWidget.__init__( self, parent )

        self.adc = adc

        self.names = []
        for i in xrange( len( adc.images[0].imageFiles ) ):
            name,path = adc.images[0].imageFiles[i]
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
        self.clusterConfiguration[ id ] = ( name, config )

    def add_supercluster(self, id, name):

        self.remove_supercluster_button.setEnabled( True )

        supercluster_widget = ClusterConfigurationWidget( id, name, self.adc )
        self.superclusters.append( supercluster_widget )
        self.connect( supercluster_widget, SIGNAL('configurationChanged'), self.on_change_configuration )

        #self.clusterConfiguration[ name ] = []
        self.clusterConfiguration.append( ( name, [] ) )

        index = self.vbox.count()
        self.vbox.insertWidget( index, supercluster_widget )

    def on_add_supercluster(self):

        id = len( self.superclusters )
        name = 'SuperCluster %d' % ( id + 1 )

        self.add_supercluster( id, name )

    def on_remove_supercluster(self):
        supercluster_widget = self.superclusters[ -1 ]
        #del self.clusterConfiguration[ supercluster_widget.name ]
        del self.clusterConfiguration[ -1 ]
        self.vbox.removeWidget( supercluster_widget )
        supercluster_widget.close()
        del self.superclusters[ -1 ]
        if len( self.superclusters ) <= 1:
            self.remove_supercluster_button.setEnabled( False )

    def build_widget(self):

        self.vbox = QVBoxLayout()

        add_supercluster_button = QPushButton( 'Add SuperCluster' )
        self.connect( add_supercluster_button, SIGNAL('clicked()'), self.on_add_supercluster )
        self.remove_supercluster_button = QPushButton( 'Remove SuperCluster' )
        self.connect( self.remove_supercluster_button, SIGNAL('clicked()'), self.on_remove_supercluster )
        self.remove_supercluster_button.setEnabled( False )
        
        hbox = QHBoxLayout()
        hbox.addWidget( add_supercluster_button )
        hbox.addWidget( self.remove_supercluster_button )
        
        groupBox = QGroupBox( 'SuperCluster configuration' )
        groupBox.setLayout( self.vbox )

        vbox = QVBoxLayout()
        vbox.addWidget( groupBox )

        vbox.addLayout( hbox )

        self.setLayout( vbox )

        self.on_add_supercluster()

