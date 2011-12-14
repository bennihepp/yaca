# -*- coding: utf-8 -*-

"""
channel_description_widgets.py -- Configuration widget for channel descriptions.
"""

# This software is distributed under the FreeBSD License.
# See the accompanying file LICENSE for details.
# 
# Copyright 2011 Benjamin Hepp

from PyQt4.QtCore import *
from PyQt4.QtGui import *

from ..core import parameter_utils as utils


class ChannelMappingCombo(QComboBox):

    __pyqtSignals__ = ('mappingChanged',)

    def __init__(self, channel, names, descrs, parent=None):

        QComboBox.__init__( self, parent )
        
        self.channel = channel
        
        self.addItem( "Don't use", '' )

        for i in xrange( len( names ) ):
            name = names[i]
            descr = descrs[i]
            self.addItem( descr, name )

        self.connect( self, SIGNAL('currentIndexChanged(int)'), self.on_current_index_changed )

        self.setCurrentIndex( 0 )

    def update_names(self, names):
        for i in xrange( 1, self.count() ):
            self.setItemText( i, names[ i - 1 ] )

    def set_current_name(self, name):
        for i in xrange( self.count() ):
            item_name = str( self.itemData( i ).toString() )
            if name == item_name:
                self.setCurrentIndex( i )
                break

    def on_current_index_changed(self, i):
            name = str( self.itemData( i ).toString() )
            if len( name ) <= 0:
                    name = None
            self.emit( SIGNAL('mappingChanged'), self.channel, name )



class ChannelDescriptionWidget(QWidget):

    __pyqtSignals__ = ('change',)

    def __init__(self, name, parent=None):

        QWidget.__init__( self, parent )
        
        self.name = name
        
        self.subwidgets = []

        self.build_widget()

    def on_lineedit_changed(self, value):
        descr = str( value )
        self.emit( SIGNAL('change'), self.name, descr )

    def setText(self, text):
        self.lineedit.setText( text )

    def build_widget(self):
    
        idlabel = QLabel( 'ID: %s' % self.name )
        namelabel = QLabel( 'Name:' )
        self.lineedit = QLineEdit()
        self.lineedit.setText( self.name )
        
        hbox = QHBoxLayout()
        
        hbox.addWidget( idlabel )
        hbox.addWidget( namelabel )
        hbox.addWidget( self.lineedit, 1 )
        
        self.connect( self.lineedit, SIGNAL('textChanged(QString)'), self.on_lineedit_changed )
        
        self.setLayout( hbox )


class ChannelDescriptionTab(QWidget):

    __OBJECT_NAME = 'ChannelDescription'

    def __init__(self, pdc, parent=None):

        QWidget.__init__( self, parent )

        self.pdc = pdc
        
        self.names = []
        for i in xrange( len( pdc.images[0].imageFiles ) ):
            name,path = pdc.images[0].imageFiles[i]
            self.names.append( name )
        
        self.channelDescription = {}
        self.channelMapping = {}

        self.cdws = []
        self.combos = []
        self.labels = []
        self.hboxes = []

        self.build_widget()

        utils.register_object( self.__OBJECT_NAME )
        utils.register_attribute( self.__OBJECT_NAME, 'channelMappingAndDescription', self.getChannelMappingDescr, self.setChannelMappingDescr )


    def cmp_channels( self, c1, c2 ):
        if c1 == c2:            return 0
        if c1 == 'R': return -1
        if c2 == 'R': return +1
        if c1 == 'G': return -1
        if c2 == 'G': return +1
        if c1 == 'B': return -1
        if c2 == 'B': return +1
        return cmp( c1, c2 )

    def getChannelMappingDescr(self):
        return ( self.channelDescription, self.channelMapping )
    def setChannelMappingDescr(self, value):

        while len( self.combos ) > 0:
            self.on_remove_overlay()

        while len( self.cdws ) > 0:
            self.cdws[ 0 ].hide()
            self.vbox1.removeWidget( self.cdws[ 0 ] )
            del self.cdws[ 0 ]

        if value:
            channelDescription, channelMapping = value
        else:
            channelDescription = {}
            channelMapping = {}

        names = channelDescription.keys()
        descrs = channelDescription.values()

        self.channelDescription = {}
        self.channelMapping = {}

        for name,descr in channelDescription.iteritems():
            cdw = ChannelDescriptionWidget( name )
            self.cdws.append( cdw )
            self.channelDescription[ name ] = descr
            self.connect( cdw, SIGNAL('change'), self.on_change_descr )
            self.vbox1.addWidget( cdw )

        tmp_d = { 'R':'Red', 'G':'Green', 'B':'Blue' }
        for c in channelMapping:
            if c not in tmp_d:
                tmp_d[ c ] = channelMapping[ c ]
        for c in 'RGB':
            if c not in channelMapping:
                channelMapping[ c ] = None

        keys = channelMapping.keys()
        keys.sort( self.cmp_channels )
        for k in keys:
            v = channelMapping[ k ]
            if v == None:
                self.add_overlay( k, None, tmp_d[ k ] )
            else:
                name = v
                self.add_overlay( k, name, tmp_d[ k ] )

        for cdw in self.cdws:
            descr = channelDescription[ cdw.name ]
            cdw.setText( descr )


    def on_change_mapping(self, channel, name):
        if not name:
            self.channelMapping[ channel ] = None
        else:
            self.channelMapping[ channel ] = name

    def add_overlay(self, channel, name, label_text):
        self.remove_overlay_button.setEnabled( True )
        descrs = []
        for n in self.names:
            descrs.append( self.channelDescription[ n ] )
        combo = ChannelMappingCombo( channel, self.names, descrs )
        label = QLabel( label_text )
        self.labels.append( label )
        self.combos.append( combo )
        self.connect( combo, SIGNAL('mappingChanged'), self.on_change_mapping )
        if name != None:
            combo.set_current_name( name )
        hbox = QHBoxLayout()
        hbox.addWidget( label )
        hbox.addWidget( combo, 1 )
        self.hboxes.append( hbox )
        index = self.vbox2.count()
        self.vbox2.insertLayout( index - 1, hbox )
    
    def on_add_overlay(self):
        number_of_overlays = len( self.combos ) - 3
        channel = 'O%d' % ( number_of_overlays + 1 )
        label_text = 'Segmentation #' + channel[ 1: ]

        self.add_overlay( channel, None, label_text )
    
    def on_remove_overlay(self):
        self.vbox2.removeItem( self.hboxes[ -1 ] )
        del self.channelMapping[ self.combos[ -1 ].channel ]
        self.combos[ -1 ].close()
        self.labels[ -1 ].close()
        self.combos[ -1 ].close()
        del self.hboxes[ -1 ]
        del self.labels[ -1 ]
        del self.combos[ -1 ]
        if len( self.combos ) <= 3:
            self.remove_overlay_button.setEnabled( False )
    
    def on_change_descr(self, name, descr):
        self.channelDescription[ name ] = descr

        descr = []
        for name in self.names:
            descr.append( self.channelDescription[ name ] )

        for combo in self.combos:
            combo.update_names( descr )

    def build_widget(self):

        self.vbox1 = QVBoxLayout()
        
        for i in xrange( len( self.pdc.images[0].imageFiles ) ):
            name,path = self.pdc.images[0].imageFiles[i]
            cdw = ChannelDescriptionWidget( name )
            self.cdws.append( cdw )
            self.channelDescription[ name ] = name
            self.connect( cdw, SIGNAL('change'), self.on_change_descr )
            self.vbox1.addWidget( cdw )
        
        self.vbox2 = QVBoxLayout()
        for c in [ 'R', 'G', 'B' ]:

            label = QLabel()
            self.labels.append( label )
            combo = ChannelMappingCombo( c, self.names, self.names )
            self.combos.append( combo )
            self.connect( combo, SIGNAL('mappingChanged'), self.on_change_mapping )

            if c == 'R':
                label.setText( 'Red' )
                combo.setCurrentIndex( 1 )
            elif c == 'G':
                label.setText( 'Green' )
                combo.setCurrentIndex( 2 )
            elif c == 'B':
                label.setText( 'Blue' )
                combo.setCurrentIndex( 3 )
            else:
                label.setText( 'Segmentation #' + c[ 1: ] )

            hbox = QHBoxLayout()
            hbox.addWidget( label )
            hbox.addWidget( combo, 1 )
            self.hboxes.append( hbox )
            self.vbox2.addLayout( hbox )
        
        add_overlay_button = QPushButton( 'Add overlay channel' )
        self.connect( add_overlay_button, SIGNAL('clicked()'), self.on_add_overlay )
        self.remove_overlay_button = QPushButton( 'Remove overlay channel' )
        self.connect( self.remove_overlay_button, SIGNAL('clicked()'), self.on_remove_overlay )
        self.remove_overlay_button.setEnabled( False )
        
        hbox = QHBoxLayout()
        hbox.addWidget( add_overlay_button )
        hbox.addWidget( self.remove_overlay_button )
        
        self.vbox2.addLayout( hbox )
        
        groupBox1 = QGroupBox( 'Channel description' )
        groupBox1.setLayout( self.vbox1 )
        
        groupBox2 = QGroupBox( 'Channel mapping' )
        groupBox2.setLayout( self.vbox2 )
        
        vbox = QVBoxLayout()
        vbox.addWidget( groupBox1 )
        vbox.addWidget( groupBox2 )
        
        self.setLayout( vbox )
