import sys, os, random
from PyQt4.QtCore import *
from PyQt4.QtGui import *
from PyQt4.QtOpenGL import *

from ..main_gui import TRY_OPENGL

import numpy
import cStringIO
import bz2
import Image
import ImageChops
import time



TMP_IMAGE_FILENAME_TEMPLATE = '/dev/shm/adc-tmp-image-file-%s-%s-%s.tiff'



class GalleryFeatureSelector(QScrollArea):

    __pyqtSignals__ = ('selectedFeaturesChanged',)

    def __init__(self, featureDescription, parent=None):

        QScrollArea.__init__( self, parent )

        self.featureDescription = featureDescription

        self.selectedFeatures = []

        self.build_widget()

    def on_feature_selected(self, id):

        if id in self.selectedFeatures:
            self.selectedFeatures.remove( id )
        else:
            self.selectedFeatures.append( id )

        self.selectedFeatures.sort()

        self.emit( SIGNAL('selectedFeaturesChanged'), self.selectedFeatures )

    def build_widget(self):

        self.buttonGroup = QButtonGroup()
        self.buttonGroup.setExclusive( False )

        featureNames = self.featureDescription.keys()
        featureNames.sort()

        vbox = QVBoxLayout()

        for featureName in featureNames:
            featureId = self.featureDescription[ featureName ]
            featureButton = QCheckBox( featureName )
            self.buttonGroup.addButton( featureButton, featureId )
            vbox.addWidget( featureButton )

        widget = QWidget()
        widget.setLayout( vbox )

        self.setWidget( widget )

        self.connect( self.buttonGroup, SIGNAL('buttonClicked(int)'), self.on_feature_selected )



class GalleryChannelAdjuster(QGroupBox):

    __pyqtSignals__ = ('adjust',)

    def __init__(self, channelName, channelDescription, parent=None):
        self.channelName = channelName
        self.channelDescription = channelDescription
        QGroupBox.__init__( self, '%s [%s]' % (channelDescription, channelName), parent )
        self.setCheckable( True )

        self.subwidgets = []
        self.build_widget()

    def on_valueChanged(self, value):
        self.black_spinBox.setMaximum( self.white_spinBox.value() )
        self.black_slider.setMaximum( self.white_spinBox.value() )
        self.white_spinBox.setMinimum( self.black_spinBox.value() )
        self.white_slider.setMinimum( self.black_spinBox.value() )
        self.on_adjust()

    def on_stateChanged(self, checked, emitSignal=True):
        if ( checked ):
            for w in self.subwidgets:
                w.show()
        else:
            for w in self.subwidgets:
                w.hide()

        if emitSignal:
            self.on_adjust()

    def on_binaryChanged(self, checked):
        if self.isChecked():

            """if ( checked ):
                for w in [ self.black_spinBox, self.black_slider, self.white_spinBox, self.white_slider ]:
                    w.setEnabled( False )
            else:
                for w in [ self.black_spinBox, self.black_slider, self.white_spinBox, self.white_slider ]:
                    w.setEnabled( True )"""

        self.on_adjust()

    def on_invertChanged(self, checked):
        self.on_adjust()

    def on_adjust(self):
        checked = self.isChecked()
        black = self.black_spinBox.value()
        white = self.white_spinBox.value()
        brightness = self.brightness_spinBox.value()
        invert = bool( self.invert_checkBox.isChecked() )
        binary = bool( self.binary_checkBox.isChecked() )
        values = ( black, white, brightness, invert, binary )
        self.emit( SIGNAL('adjust'), self.channelName, checked, values )

    def setActive(self, active, emitSignal=True):
        self.setChecked( active )
        self.on_stateChanged( active, emitSignal )
        if emitSignal:
            self.on_adjust()

    def isActive(self):
        return self.isChecked()

    def brightness_spinBox_changed(self, value):
        self.brightness_slider.setValue( int( value * 100.0 + 0.5 ) )

    def brightness_slider_changed(self, value):
        self.brightness_spinBox.setValue( value / 100.0 )

    def build_widget(self):

        self.connect( self, SIGNAL('clicked(bool)'), self.on_stateChanged )

        invert_label = QLabel( 'Invert' )
        self.invert_checkBox = QCheckBox()
        self.invert_checkBox.setChecked( False )
        self.connect( self.invert_checkBox, SIGNAL('stateChanged(int)'), self.on_invertChanged )

        binary_label = QLabel( 'Binary' )
        self.binary_checkBox = QCheckBox()
        self.binary_checkBox.setChecked( False )
        self.connect( self.binary_checkBox, SIGNAL('stateChanged(int)'), self.on_binaryChanged )

        binary_invert_hbox = QHBoxLayout()
        binary_invert_hbox.addWidget( self.invert_checkBox )
        binary_invert_hbox.addWidget( invert_label )
        binary_invert_hbox.addWidget( self.binary_checkBox )
        binary_invert_hbox.addWidget( binary_label )

        self.brightness_spinBox = QDoubleSpinBox()
        self.brightness_spinBox.setMinimum( 0.0 )
        self.brightness_spinBox.setMaximum( 10.0 )
        self.brightness_spinBox.setValue( 1.0 )
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setMinimum( 0 )
        self.brightness_slider.setMaximum( 1000 )
        self.brightness_slider.setValue( 100 )
        self.brightness_slider.setTracking( False )
        QObject.connect( self.brightness_spinBox, SIGNAL('valueChanged(double)'), self.brightness_spinBox_changed )
        QObject.connect( self.brightness_slider, SIGNAL('valueChanged(int)'), self.brightness_slider_changed )
        self.connect( self.brightness_spinBox, SIGNAL('valueChanged(double)'), self.on_valueChanged )

        brightness_hbox = QHBoxLayout()
        brightness_label = QLabel('Brightness:')
        brightness_hbox.addWidget( brightness_label )
        brightness_hbox.addWidget(self.brightness_spinBox)
        brightness_hbox.addWidget(self.brightness_slider)

        self.black_spinBox = QSpinBox()
        self.black_spinBox.setMinimum( 0 )
        self.black_spinBox.setMaximum( 255 )
        self.black_spinBox.setValue( 0 )
        self.black_slider = QSlider(Qt.Horizontal)
        self.black_slider.setMinimum( 0 )
        self.black_slider.setMaximum( 255 )
        self.black_slider.setValue( 0 )
        self.black_slider.setTracking( False )
        QObject.connect( self.black_spinBox, SIGNAL('valueChanged(int)'), self.black_slider.setValue )
        QObject.connect( self.black_slider, SIGNAL('valueChanged(int)'), self.black_spinBox.setValue )
        self.connect( self.black_spinBox, SIGNAL('valueChanged(int)'), self.on_valueChanged )

        black_hbox = QHBoxLayout()
        black_label = QLabel('Black:')
        black_hbox.addWidget( black_label )
        black_hbox.addWidget(self.black_spinBox)
        black_hbox.addWidget(self.black_slider)

        self.white_spinBox = QSpinBox()
        self.white_spinBox.setMinimum( 0 )
        self.white_spinBox.setMaximum( 255 )
        self.white_spinBox.setValue( 255 )
        self.white_slider = QSlider(Qt.Horizontal)
        self.white_slider.setMinimum( 0 )
        self.white_slider.setMaximum( 255 )
        self.white_slider.setValue( 255 )
        self.white_slider.setTracking( False )
        QObject.connect( self.white_spinBox, SIGNAL('valueChanged(int)'), self.white_slider.setValue )
        QObject.connect( self.white_slider, SIGNAL('valueChanged(int)'), self.white_spinBox.setValue )
        self.connect( self.white_spinBox, SIGNAL('valueChanged(int)'), self.on_valueChanged )

        white_hbox = QHBoxLayout()
        white_label = QLabel('White:')
        white_hbox.addWidget( white_label )
        white_hbox.addWidget(self.white_spinBox)
        white_hbox.addWidget(self.white_slider)

        self.vbox = QVBoxLayout()
        self.vbox.addLayout( binary_invert_hbox )
        self.vbox.addLayout( brightness_hbox )
        self.vbox.addLayout( black_hbox )
        self.vbox.addLayout( white_hbox )

        self.subwidgets.append( brightness_label )
        self.subwidgets.append( self.brightness_spinBox )
        self.subwidgets.append( self.brightness_slider )
        self.subwidgets.append( black_label )
        self.subwidgets.append( self.black_spinBox )
        self.subwidgets.append( self.black_slider )
        self.subwidgets.append( white_label )
        self.subwidgets.append( self.white_spinBox )
        self.subwidgets.append( self.white_slider )

        self.setLayout( self.vbox )



GalleryPixmapAreaBaseClass = QWidget
if TRY_OPENGL:
    GalleryPixmapAreaBaseClass = QGLWidget

class GalleryPixmapArea(GalleryPixmapAreaBaseClass):

    __pyqtSignals__ = ('pixmapSelected',)

    PIXMAP_SPACING = 8

    TEXT_VERTICAL_OFFSET = 1
    TEXT_HORIZONTAL_OFFSET = 1
    TEXT_VERTICAL_SPACING = 2
    TEXT_FONT_FAMILY = 'Arial'
    TEXT_FONT_SIZE = 10

    def __init__(self, parent=None):
        GalleryPixmapAreaBaseClass.__init__( self, parent )
        #self.textures = []
        self.pixmaps = []
        self.pixmap_texts = []

    """def convert_pixmaps_to_textures(self, pixmaps):
        textures = []
        for guint in self.textures:
            self.deleteTexture( guint )
        for i in xrange( len( pixmaps ) ):
            if pixmaps[ i ] != None:
                guint = self.bindTexture( pixmaps[ i ] )
            else:
                guint = None
            textures.append( guint )
        return textures"""

    def mouseDoubleClickEvent(self, event):
        btn = event.buttons()
        if btn == Qt.LeftButton:
            x,y = event.x(),event.y()
            x -= self.PIXMAP_SPACING
            y -= self.PIXMAP_SPACING
            column = x / ( self.PIXMAP_SPACING + self.pixmap_width )
            row = y / ( self.PIXMAP_SPACING + self.pixmap_height )
            index = row * self.columns + column
            self.emit( SIGNAL('pixmapSelected'), index, row, column, event )

    def set_pixmaps_and_texts(self, rows, columns, pixmaps, focus_index, pixmap_texts):
        self.rows = rows
        self.columns = columns
        self.focus_index = focus_index
        self.pixmap_width = pixmaps[ 0 ].width()
        self.pixmap_height = pixmaps[ 0 ].height()
        self.pixmaps = pixmaps
        self.pixmap_texts = pixmap_texts
        #self.textures = self.convert_pixmaps_to_textures( pixmaps )
        width =  self.PIXMAP_SPACING + self.columns * ( self.PIXMAP_SPACING + pixmaps[ 0 ].width() )
        height = self.PIXMAP_SPACING +    self.rows * ( self.PIXMAP_SPACING + pixmaps[ 0 ].height() )
        #print 'width=%d, height=%d' % (width,height)
        self.setMinimumSize( width, height )
        self.setMaximumSize( width, height )
        self.resize( width, height )
        #print 'updating'
        self.update()

    def set_pixmaps(self, rows, columns, pixmaps, focus_index=-1):
        self.set_pixmaps_and_texts( rows, columns, pixmaps, focus_index, None )

    """def draw_texture(self, row, column, texture):
        left = self.PIXMAP_SPACING + column * ( self.PIXMAP_SPACING + self.pixmap_width )
        top =  self.PIXMAP_SPACING +    row * ( self.PIXMAP_SPACING + self.pixmap_height )

        rect = QRectF( left, top, left + self.pixmap_width, top + self.pixmap_height )

        self.drawTexture( rect, texture )"""

    def draw_pixmap(self, painter, row, column, pixmap, focused=False):

        left = self.PIXMAP_SPACING + column * ( self.PIXMAP_SPACING + self.pixmap_width )
        top =  self.PIXMAP_SPACING +    row * ( self.PIXMAP_SPACING + self.pixmap_height )

        point = QPoint( left, top )

        painter.drawPixmap( point, pixmap )

        if focused:

            penWidthF = 2.0
            pen = QPen( QColor( 0, 255, 255 ) )
            pen.setWidthF( penWidthF )
            painter.setPen( pen )

            left -= self.PIXMAP_SPACING / 2 + penWidthF / 2
            top -= self.PIXMAP_SPACING / 2 + penWidthF / 2
            width = self.pixmap_width + self.PIXMAP_SPACING
            height = self.pixmap_height + self.PIXMAP_SPACING
            xradius = width / 20
            yradius = height / 20

            #painter.drawRect( QRect( left, top, left + width, top + height ) )
            painter.drawRoundedRect( QRectF( left, top, left + width, top + height ), xradius, yradius )

    def draw_texts(self, painter, row, column, texts):

        if len( texts ) > 0:

            painter.setPen( QColor( 255,255,255 ) )
            painter.setFont( QFont( self.TEXT_FONT_FAMILY, self.TEXT_FONT_SIZE ) )

            left = self.PIXMAP_SPACING + column * ( self.PIXMAP_SPACING + self.pixmap_width )
            top =  self.PIXMAP_SPACING +    row * ( self.PIXMAP_SPACING + self.pixmap_height )
            width = self.pixmap_width
            height = self.pixmap_height

            x = self.TEXT_HORIZONTAL_OFFSET
            y = self.TEXT_VERTICAL_OFFSET

            for text in texts:

                rect = QRect(
                    left + x,
                    top + y,
                    left + width,
                    top + y + height
                )

                boundingRect = painter.boundingRect( rect, Qt.TextSingleLine | Qt.AlignLeft | Qt.AlignTop, text )

                y += boundingRect.height() + self.TEXT_VERTICAL_SPACING

                painter.drawText( boundingRect, Qt.TextSingleLine | Qt.AlignLeft | Qt.AlignTop, text )

    def paintEvent(self, event):
        #print 'paintEvent'
        p = QPainter()
        p.begin( self)

        p.setRenderHint( QPainter.Antialiasing )

        color = QApplication.palette().color( QPalette.Background )
        brush = QBrush( color )
        rect = QRect( 0, 0, self.width(), self.height() )
        p.fillRect( rect, brush )

        for r in xrange( self.rows ):

            for c in xrange( self.columns ):

                index = r * self.columns + c

                if index < len( self.pixmaps ):

                    pixmap = self.pixmaps[ index ]
                    if pixmap != None:
                        self.draw_pixmap( p, r, c, pixmap, self.focus_index == index )

                    if self.pixmap_texts and len( self.pixmap_texts ) > index:
                        texts = self.pixmap_texts[ index ]
                        if texts != None:
                            self.draw_texts( p, r, c, texts )

        p.end()

        """for r in xrange( self.rows ):

            for c in xrange( self.columns ):

                texture = self.textures[ r * self.columns + c ]
                if texture != None:
                    self.draw_texture( r, c, texture )"""



class GalleryWindow(QWidget):

    DEFAULT_PIXMAP_SIZE = 128
    MIN_PIXMAP_SIZE = 64
    MAX_PIXMAP_SIZE = 256

    def __init__(self, featureDescription, channelMapping, channelDescription, singleImage=False, parent=None):
        super(QWidget,self).__init__(parent)
        self.setWindowTitle('APC Gallery')

        self.singleImage = singleImage

        self.tmp_image_filename = TMP_IMAGE_FILENAME_TEMPLATE
        self.tmp_image_filename_rnd = numpy.random.randint(sys.maxint)

        self.selectionIds = None

        self.pixmaps = []
        #self.widgets = []
        self.imageCache = {}

        self.featureDescription = featureDescription

        self.selectedFeatures = []

        self.channelMapping = channelMapping
        self.channelDescription = channelDescription
        self.channelAdjustment = {}

        self.random = False

        #self.setSizePolicy( QSizePolicy( QSizePolicy.Fixed, QSizePolicy.Fixed ) )

        #self.create_menu()
        self.create_frame()

    def cmp_channels( self, c1, c2 ):
        if c1 == c2:
            return 0
        if c1 == 'R': return -1
        if c2 == 'R': return +1
        if c1 == 'G': return -1
        if c2 == 'G': return +1
        if c1 == 'B': return -1
        if c2 == 'B': return +1
        return cmp( c1, c2 )

    def on_selection_changed(self, focusId, selectionIds, pixmapFactory, featureFactory):

        self.focusId = focusId
        self.selectionIds = selectionIds

        self.idMapping = []
        for i in xrange( len( selectionIds ) ):
            self.idMapping.append( i )

        if self.random:
            self.randomize_mapping()

        self.pixmapFactory = pixmapFactory
        self.featureFactory = featureFactory

        self.focus_i = -1
        for i in xrange( len( selectionIds ) ):
            if selectionIds[i] == focusId:
                self.focus_i = i
                break

        self.start_i = max( 0, self.focus_i - self.rows * self.columns / 2 )
        self.stop_i = min( self.start_i + self.rows * self.columns, len( selectionIds ) )

        self.img_comp = None

        self.on_reload_images( self.start_i, self.stop_i, self.focus_i )

    def concatenate_imgs(self, img1, img2, mode='side-by-side'):
        img_mode = img1.mode
        img_size = img1.size
        new_img_size = ( 2*img_size[0], 2*img_size[1] )
        img3 = Image.new( img_mode, new_img_size )
        img3.paste( img1, ( 0,0 ) )
        img3.paste( img2, ( img_size[0],0 ) )
        img3.paste( img2, ( 0,img_size[1] ) )
        img3.paste( img1, ( img_size[0],img_size[1] ) )
        return img3

    def compress_and_measure_img(self, img, format='JPEG', **kwargs):
        if format == 'JPEG' or 'PNG':
            sio = cStringIO.StringIO()
            img.save( sio, format, **kwargs)
            s = sio.getvalue()
            sio.close()
            s = bz2.compress( s )
            return float( len( s ) )
        elif format == 'pgm':
            sio = cStringIO.StringIO()
            arr = numpy.array( img.getdata() )
            for c in arr:
                sio.write('%d ') % c
            s = sio.getvalue()
            sio.close()
            s = bz2.compress( s )
            return float( len( s ) )

    def load_cell_image(self, imageId):
        img = self.pixmapFactory.createImage(
                                imageId,
                                -1,
                                -1,
                                self.pixmap_width,
                                self.pixmap_height,
                                self.channelAdjustment,
                                self.color,
                                self.tmp_image_filename % ( str( self.tmp_image_filename_rnd ), str( imageId ), str( time.time() ) ),
                                self.imageCache,
                                int( imageId )
        )
        name = 'CellsObjects'
        img_mask = self.pixmapFactory.createImageMask(
                                imageId,
                                name,
                                -1,
                                -1,
                                self.pixmap_width,
                                self.pixmap_height,
                                self.tmp_image_filename % ( str( self.tmp_image_filename_rnd ), str( imageId ), str( time.time() ) ),
                                self.imageCache,
                                int( imageId )
        )
        return ImageChops.darker( img, img_mask )

    def comp_imgs(self, id1, ids, format='JPEG', **kwargs):
        img1 = self.load_cell_image( int( id1 ) )
        imgs = []
        for id2 in ids:
            img2 = self.load_cell_image( int( id2 ) )
            imgs.append( img2 )

        C1 = self.compress_and_measure_img( img1, format, **kwargs )
        C2s = []
        for img2 in imgs:
            C2s.append( self.compress_and_measure_img( img2, format, **kwargs) )
        C2s = numpy.array( C2s )
        C12s = []
        for img2 in imgs:
            img3 = self.concatenate_imgs( img1, img2 )
            C12 = self.compress_and_measure_img( img3, format, **kwargs )
            img4 = self.concatenate_imgs( img2, img1 )
            C12 += self.compress_and_measure_img( img4, format, **kwargs )
            C12 /= 4.0
            C12s.append( C12 )
        C12s = numpy.array( C12s )

        self.gw = GalleryWindow( self.featureDescription, self.channelMapping, self.channelDescription )
        from gui_utils import CustomPixmapFactory
        pixmapFactory = CustomPixmapFactory( imgs )
        self.gw.on_selection_changed( -1, range( len( imgs ) ), pixmapFactory, None )
        self.gw.show()

        NCDs = []
        for i in xrange( len( C2s ) ):
            NCD_numerator = C12s[ i ] - min( C1, C2s[ i ] )
            NCD_denominator = max( C1, C2s[ i ] )
            NCDs.append( NCD_numerator / NCD_denominator )

        for i in xrange( len( NCDs ) ):
            print '%d: %f' % (i, NCDs[i])

        NCDs = numpy.array( NCDs )

    def on_pixmap_selected(self, index, row, column, event):
        i = self.start_i + index
        pixmapId = self.idMapping[ i ]
        if self.random and index == 0 and self.focus_i >= 0:
            pixmapId = self.focus_i
        id1 = self.selectionIds[ pixmapId ]
        ids = []
        for i in xrange( self.start_i, self.stop_i ):
            pixmapId = self.idMapping[ i ]
            if self.random and i == self.start_i and self.focus_i >= 0:
                pixmapId = self.focus_i
            ids.append( self.selectionIds[ pixmapId ] )
        self.comp_imgs(id1, ids)

    def on_reload_images(self, start_i, stop_i, focus_i=-1):

        if self.selectionIds == None:
            return

        if stop_i > start_i:
            self.progressbar.setRange( start_i, stop_i - 1 )
        else:
            self.progressbar.setRange( start_i, start_i + 1 )

        #self.progressbar.setValue( start_i )
        self.progressbar.setTextVisible( True )
        self.progressbar.setFormat( 'Loading images - %p%' )

        if self.random:
            randomStr = ' random'
        else:
            randomStr = ''

        if self.singleImage:
            self.statuslabel.setText( 'Showing%s image %d out of %d images' % ( randomStr, start_i+1, len(self.selectionIds) ) )
        else:
            self.statuslabel.setText( 'Showing%s cell %d - %d out of %d cells' % ( randomStr, start_i+1, stop_i, len(self.selectionIds) ) )

        #for w in self.widgets:
        #    w.clear()

        oldCacheIds = self.imageCache.keys()

        del self.pixmaps[ : ]

        pixmap_texts = []

        focus_index = -1

        for i in xrange( start_i, stop_i ):

            self.progressbar.setValue( i )

            try:
                pixmapId = self.idMapping[ i ]
                if self.random and ( i == start_i ) and ( focus_i >= 0 ):
                    pixmapId = focus_i
                    focus_index = len( self.pixmaps )

                cacheId = int( self.selectionIds[ pixmapId ] )
                if cacheId in oldCacheIds:
                    oldCacheIds.remove( cacheId )
                else:
                    self.imageCache[ cacheId ] = {}
        
                pix = self.pixmapFactory.createPixmap(
                                self.selectionIds[ pixmapId ],
                                -1,
                                -1,
                                self.pixmap_width,
                                self.pixmap_height,
                                self.channelAdjustment,
                                self.color,
                                self.tmp_image_filename % ( str( self.tmp_image_filename_rnd ), str( cacheId ), str( time.time() ) ),
                                self.imageCache
                )

                #print id(pix)

                if pix:
                    self.pixmaps.append(pix)

                else:
                    raise Exception( 'Error when creating pixmap' )

                    #self.widgets[i-start_i].setText('Label #%d' % (i))
                    #self.widgets[i-start_i].setPixmap(pix)
                    #if pixmapId == self.focus_i:
                    #    self.widgets[i-start_i].setFrameStyle(QFrame.Box)
                    #    self.widgets[i-start_i].setLineWidth(2)
                    #else:
                    #    self.widgets[i-start_i].setFrameStyle(QFrame.NoFrame)

                #else:
                    #self.widgets[i-start_i].setText('No image files available')
                    #self.pixmaps.append(None)
                    #self.widgets[i-start_i].setFrameStyle(QFrame.Box)
                    #self.widgets[i-start_i].setLineWidth(2)

                if len( self.selectedFeatures ) > 0:

                    texts = []

                    for feature in self.selectedFeatures:

                        j = self.featureDescription.values().index( feature )

                        featureName = self.featureDescription.keys()[ j ]
                        if self.singleImage:
                            text = '%s=%s' % ( featureName, str( self.featureFactory.createFeatureText( self.selectionIds[ pixmapId ], feature ) ) )
                        else:
                            text = str( self.featureFactory.createFeatureText( self.selectionIds[ pixmapId ], feature ) )

                        texts.append( text )

                    if len( texts ) > 0:
                        pixmap_texts.append( texts )
                    else:
                        pixmap_texts.append( None )

                """
            if channels:
                try:
                    print '<0>'
                    rect = rectSelector( selectionIds[i] )
                    if channels.has_key('R'):
                        img = Image.open( channels['R'] )
                        img_red = img.crop( rect )
                        img_red = image_convert_16_to_8_bit( img_red )
                        del img
                    else:
                        img_red = Image.new( 'L', ( (rect[2]-rect[0]) , (rect[3]-rect[1]) ) )
                    if channels.has_key('G'):
                        img = Image.open( channels['G'] )
                        img_green = img.crop( rect )
                        img_green = image_convert_16_to_8_bit( img_green )
                        del img
                    else:
                        img_green = Image.new( 'L', ( (rect[2]-rect[0]) , (rect[3]-rect[1]) ) )
                    if channels.has_key('B'):
                        img = Image.open( channels['B'] )
                        img_blue = img.crop( rect )
                        img_blue = image_convert_16_to_8_bit( img_blue )
                        del img
                    else:
                        img_blue = Image.new( 'L', ( (rect[2]-rect[0]) , (rect[3]-rect[1]) ) )
                    mode = 'RGB'
                    img = Image.merge( mode, (img_red,img_green,img_blue) )
                    img_size = img.size
                    del img_red,img_green,img_blue

                    #i1 = Image.open(filename)
                    #print '<1>'
                    #i2 = i1.crop(rectSelector(selectionIds[i]))
                    print '<2>'
                    arr = numpy.array(img.getdata())
                    del img
                    print '<3>'
                    tmp_str = struct.pack('@%dh' % len(arr), *arr)
                    print '<4>'
                    arr = numpy.array(struct.unpack('@%dH' % len(arr), tmp_str))
                    del tmp_str
                    print '<5>'
                    arr = arr - 2**15
                    arr = arr * (2**8-1.0)/(2**12-1.0)
                    #a3 = a3 >> 4
                    print '<6>'
                    tmp_str = struct.pack('@%dB' % len(arr), *arr)
                    print '<7>'
                    img = Image.fromstring('L',img_size,tmp_str,"raw","L",0,1)
                    del tmp_str
                    print '<8>'
                    img.save(self.tmp_image_filename)
                    print '<9>'
                    del img
                    #qtimg = QImage('/home/benjamin/tmp.tif')
                    #qtimg.save('/home/benjamin/tmp_img.tif')

                    #i1 = libtiff.TIFF.open(filename)
                    #a1 = i1.read_image()
                    #a1 = a1 - 2**15
                    #a1 = a1 * ( 2**8 - 1.0 ) / ( 2**12 - 1.0 )


                    #tmp = Image.open(filename)
                    #img = tmp.crop(rectSelector(selectionIds[i]))
                    #img.save('/home/benjamin/tmp.tiff')
                    #img2 = img.convert("L")
                    #img2.save('/home/benjamin/tmp2.tiff')
                    #del img
                    #del tmp
                    #qtimg = ImageQt.ImageQt(img2)
                    
                    #rect = rectSelector(selectionIds[i])
                    #tmp = QImage(filename)
                    #qtimg = tmp.copy(rect[0], rect[1], rect[2]-rect[0], rect[3]-rect[1])
                    #del tmp
    
                    #pix = QPixmap.fromImage(qtimg)
                    #pix.save('/home/benjamin/tmp_pix.png')
                    #del qtimg
    
                    #rect = rectSelector(selectionIds[i])
                    #tmp = QPixmap(filename)
                    #pix = tmp.copy(rect[0], rect[1], rect[2]-rect[0], rect[3]-rect[1])
                    #del tmp
    
                    pix = QPixmap(self.tmp_image_filename)
                    print '<10>'
                    self.pixmaps.append(pix)
                    print '<11>'
                    self.widgets[i-start_i].setPixmap(pix)
                    print '<12>'"""

            except IOError:
                #self.widgets[i-start_i].setText('Invalid filenames')
                self.pixmaps.append(None)
                #self.widgets[i-start_i].setFrameStyle(QFrame.Box)
                #self.widgets[i-start_i].setLineWidth(2)
                raise

            except:
                #self.widgets[i-start_i].setText('Invalid image files')
                self.pixmaps.append(None)
                #self.widgets[i-start_i].setFrameStyle(QFrame.Box)
                #self.widgets[i-start_i].setLineWidth(2)
                raise
            finally:
                if os.path.isfile( self.tmp_image_filename ):
                    os.remove( self.tmp_image_filename )

        self.progressbar.setValue( self.progressbar.maximum() )

        if len( pixmap_texts ) > 0:
            self.pixmaparea.set_pixmaps_and_texts( self.rows, self.columns, self.pixmaps, focus_index, pixmap_texts )
        else:
            self.pixmaparea.set_pixmaps( self.rows, self.columns, self.pixmaps, focus_index )

        for cacheId in oldCacheIds:
            del self.imageCache[ cacheId ]

        self.progressbar.setFormat( 'Images loaded' )

    def on_previous(self):
        if self.start_i > 0:
            self.stop_i = self.start_i
            self.start_i = max( 0, self.stop_i - self.rows * self.columns )
            self.on_reload_images( self.start_i, self.stop_i, self.focus_i )
    def on_next(self):
        if self.stop_i + 1 < len( self.selectionIds ):
            self.start_i = self.stop_i
            self.stop_i = min( self.start_i + self.rows * self.columns, len( self.selectionIds ) )
            self.on_reload_images( self.start_i, self.stop_i, self.focus_i )

    """def calculate_pixmap_size(self):
        size = self.pixmaparea.size()
        if self.columns > 1:
            self.pixmap_width = ( size.width() / self.columns - self.GRID_SPACING / (self.columns-1) )
        else:
            self.pixmap_width = size.width() / self.columns
        #if self.pixmap_width % 2 != 0:
        #    self.pixmap_width -= 1
        if self.rows > 1:
            self.pixmap_height = ( size.height() / self.rows - self.GRID_SPACING / (self.rows-1)  )
        else:
            self.pixmap_height = size.height() / self.rows
        #if self.pixmap_height % 2 != 0:
        #    self.pixmap_height -= 1

    def resizeEvent(self, event):

        print 'resizeEvent'

        self.calculate_pixmap_size()

        self.pixmaparea.resize( ( self.pixmap_width * self.columns + self.GRID_SPACING * (self.columns-1) + 2 * self.LABEL_MARGIN * self.columns ),
                                ( self.pixmap_height * self.rows + self.GRID_SPACING * (self.rows-1) + 2 * self.LABEL_MARGIN * self.rows )
        )

        for label in self.widgets:
            label.resize( self.pixmap_width + 2 * self.LABEL_MARGIN, self.pixmap_height + 2 * self.LABEL_MARGIN )

        if self.selectionIds != None:
            self.on_reload_images( self.start_i, self.stop_i, self.focus_i )"""

    def on_size_changed(self, size):
        self.pixmap_width = size
        self.pixmap_height = size

        #for label in self.widgets:
        #    label.resize( self.pixmap_width + self.LABEL_MARGIN, self.pixmap_height + self.LABEL_MARGIN )
        #    label.setMargin( self.LABEL_MARGIN )

        self.resize( self.minimumSize() )

        if self.selectionIds != None:
            self.stop_i = min( self.start_i + self.rows * self.columns, len( self.selectionIds ) )
            self.on_reload_images( self.start_i, self.stop_i, self.focus_i )

    def on_row_or_column_changed(self):
        #for widget in self.widgets:
        #    self.grid.removeWidget(widget)
        #    widget.clear()
        #del self.widgets[:]

        #self.pixmaparea.resize( ( self.pixmap_width * self.columns + self.GRID_SPACING * (self.columns-1) ),
        #                        ( self.pixmap_height * self.rows + self.GRID_SPACING * (self.rows-1) )
        #)

        """for i in xrange(self.rows):
            #self.setRowMinimumHeight(i, self.label_height)
            for j in xrange(self.columns):
                #self.setColumnMinimumWidth(self.label_width)
                ql = QLabel()
                ql.resize( self.pixmap_width + self.LABEL_MARGIN, self.pixmap_height + self.LABEL_MARGIN )
                ql.setMargin( self.LABEL_MARGIN )
                self.widgets.append(ql)
                self.grid.addWidget(ql, i, j)"""

        self.resize( self.minimumSize() )

        if self.selectionIds != None:
            self.stop_i = min( self.start_i + self.rows * self.columns, len( self.selectionIds ) )
            self.on_reload_images( self.start_i, self.stop_i, self.focus_i )

        """#print self.minimumSizeHint()
        #print self.minimumSize()
        minSizeHint = self.minimumSizeHint()
        minSize_x = max( minSizeHint.width(), self.label_width * self.columns )
        minSize_y = max( minSizeHint.height(), self.label_height * self.rows )
        minSize = QSize(minSize_x, minSize_y)
        self.setMinimumSize(minSize)
        self.resize(minSize)
        for i in xrange(self.rows):
            #self.setRowMinimumHeight(i, self.label_height)
            for j in xrange(self.columns):
                #self.setColumnMinimumWidth(self.label_width)
                ql = QLabel()
                #ql.setFixedSize(self.label_width,self.label_height)
                ql.minimumSize = (self.label_width,self.label_height)
                self.widgets.append(ql)
                self.grid.addWidget(ql, i, j)
        self.update()
        self.on_selection_changed()
        """
    def on_row_changed(self, rows):
        self.rows = rows
        self.on_row_or_column_changed()
    def on_column_changed(self, columns):
        self.columns = columns
        self.on_row_or_column_changed()

    def randomize_mapping( self ):
        random.shuffle( self.idMapping )

    def on_randomize_mapping( self ):
        self.randomize_mapping()
        self.on_reload_images( self.start_i, self.stop_i, self.focus_i )

    def on_reset_mapping( self ):
        for i in xrange( len( self.selectionIds ) ):
            self.idMapping[ i ] = i
        self.on_reload_images( self.start_i, self.stop_i, self.focus_i )

    def on_random_changed(self, state):

        self.randomize_button.setEnabled( state )

        stateChanged = False
        if self.random != state:
            stateChanged = True
        self.random = state

        if stateChanged and self.selectionIds != None:

            if self.random:
                self.on_randomize_mapping()
            else:
                self.on_reset_mapping()


    def on_channel_adjust(self, channel, checked, values):
        channel = str(channel)
        checked = bool( checked )
        if checked:
            self.channelAdjustment[ channel ] = values
        elif self.channelAdjustment.has_key( channel ):
            del self.channelAdjustment[ channel ]

        # TODO for gray-scale
        if checked and not self.color and channel in 'RGB':
            for ca in self.channelAdjusters:
                if channel != ca.channelName and ca.channelName in 'RGB':
                    ca.setActive( False, False )
                    del self.channelAdjustment[ ca.channelName ]

        if self.selectionIds != None:
            self.on_reload_images( self.start_i, self.stop_i, self.focus_i )

    def on_selected_features_changed(self, selectedFeatures):
        self.selectedFeatures = selectedFeatures
        if self.selectionIds != None:
            self.on_reload_images( self.start_i, self.stop_i, self.focus_i )

    def on_color_changed(self, color):
        self.color = bool( color )

        # TODO for gray-scale
        if not color:
            foundActive = False
            for ca in self.channelAdjusters:
                if ca.channelName in 'RGB':
                    if foundActive:
                        ca.setActive( False, False )
                    elif ca.isActive():
                        #self.channelAdjustment = { ca.channelName : self.channelAdjustment[ ca.channelName ] }
                        for c in 'RGB':
                            if c != ca.channelName and c in self.channelAdjustment:
                                del self.channelAdjustment[ c ]
                        foundActive = True

        if self.selectionIds != None:
            self.on_reload_images( self.start_i, self.stop_i, self.focus_i )

    def create_frame(self):
        #self.grid = QGridLayout()
        #self.grid.setSpacing(self.GRID_SPACING)

        if self.singleImage:
            self.pixmap_width = -1
            self.pixmap_height = -1

        self.pixmaparea = GalleryPixmapArea()
        self.connect( self.pixmaparea, SIGNAL('pixmapSelected'), self.on_pixmap_selected )
        #self.pixmaparea.setLayout(self.grid)

        hbox1 = QHBoxLayout()

        self.previous_button = QPushButton("&Previous")
        self.connect(self.previous_button, SIGNAL('clicked()'), self.on_previous)
        hbox1.addWidget(self.previous_button)
        self.next_button = QPushButton("&Next")
        self.connect(self.next_button, SIGNAL('clicked()'), self.on_next)
        hbox1.addWidget(self.next_button)

        if not self.singleImage:
            hbox1.addWidget(QLabel("Rows:"))
            self.row_box = QSpinBox()
            self.row_box.setRange(1, 10)
            self.connect(self.row_box, SIGNAL('valueChanged(int)'), self.on_row_changed)
            hbox1.addWidget(self.row_box)
            hbox1.addWidget(QLabel("Columns:"))
            self.column_box = QSpinBox()
            self.column_box.setRange(1, 10)
            self.connect(self.column_box, SIGNAL('valueChanged(int)'), self.on_column_changed)
            hbox1.addWidget(self.column_box)
            hbox1.addWidget( QLabel( 'Size:' ) )
            self.size_spinBox = QSpinBox()
            self.size_spinBox.setMinimum( self.MIN_PIXMAP_SIZE )
            self.size_spinBox.setMaximum( self.MAX_PIXMAP_SIZE )
            self.size_slider = QSlider(Qt.Horizontal)
            self.size_slider.setMinimum( self.MIN_PIXMAP_SIZE )
            self.size_slider.setMaximum( self.MAX_PIXMAP_SIZE )
            self.size_slider.setTracking( False )
            QObject.connect( self.size_spinBox, SIGNAL('valueChanged(int)'), self.size_slider.setValue )
            QObject.connect( self.size_slider, SIGNAL('valueChanged(int)'), self.size_spinBox.setValue )
            self.connect( self.size_spinBox, SIGNAL('valueChanged(int)'), self.on_size_changed )
            self.size_spinBox.setValue( self.DEFAULT_PIXMAP_SIZE )
            hbox1.addWidget( self.size_spinBox )
            hbox1.addWidget( self.size_slider )

        hbox1.addWidget(QLabel("Random:"))
        self.random_box = QCheckBox()
        self.connect(self.random_box, SIGNAL('stateChanged(int)'), self.on_random_changed)
        self.random_box.setChecked( False )
        hbox1.addWidget(self.random_box)
        self.randomize_button = QPushButton( 'Randomize' )
        self.randomize_button.setEnabled( self.random )
        self.connect(self.randomize_button, SIGNAL('clicked()'), self.on_randomize_mapping)
        hbox1.addWidget( self.randomize_button )

        self.channelAdjusters = []

        self.color = True
        colorCheckbox = QCheckBox( 'Color' )
        self.connect( colorCheckbox, SIGNAL('stateChanged(int)'), self.on_color_changed )
        colorCheckbox.setChecked( self.color )

        vbox2 = QVBoxLayout()
        vbox2.addWidget( colorCheckbox )

        channels = self.channelMapping.keys()
        channels.sort( cmp=self.cmp_channels )
        for c in channels:
            name = self.channelMapping[ c ]
            descr = self.channelDescription[ name ]
            channelAdjuster = GalleryChannelAdjuster( c, descr )
            self.channelAdjusters.append( channelAdjuster )
            self.connect( channelAdjuster, SIGNAL('adjust'), self.on_channel_adjust )
            vbox2.addWidget( channelAdjuster )
            if c in ['R','G','B']:
                channelAdjuster.setActive( True )
            else:
                channelAdjuster.setActive( False )
        channelWidgets = QWidget()
        channelWidgets.setLayout( vbox2 )

        channelScrollArea = QScrollArea()
        channelScrollArea.setMinimumWidth( channelWidgets.minimumSizeHint().width() )
        #channelScrollArea.setHorizontalScrollBarPolicy( Qt.ScrollBarAlwaysOff )
        channelScrollArea.setWidget( channelWidgets )

        featureSelector = GalleryFeatureSelector( self.featureDescription )
        self.connect( featureSelector, SIGNAL('selectedFeaturesChanged'), self.on_selected_features_changed )

        tabWidget = QTabWidget()
        tabWidget.addTab( channelScrollArea, 'Channels' )
        tabWidget.addTab( featureSelector, 'Features' )

        hsplitter = QSplitter()
        hbox2 = QHBoxLayout()
        if self.singleImage:
            sa = QScrollArea()
            sa.setWidget( self.pixmaparea )
            hsplitter.addWidget( sa )
        else:
            hsplitter.addWidget( self.pixmaparea )
        hsplitter.setStretchFactor( 0, 1 )

        hsplitter.addWidget( tabWidget )
        hsplitter.setStretchFactor( 1, 0 )

        self.statusbar = QStatusBar()

        self.statuslabel = QLabel()
        self.statusbar.addPermanentWidget( self.statuslabel, 1 )

        self.progressbar = QProgressBar()
        self.statusbar.addPermanentWidget( self.progressbar, 1 )

        vbox = QVBoxLayout()
        vbox.addLayout(hbox1)
        vbox.addWidget(hsplitter, 1)
        vbox.addWidget(self.statusbar)

        self.setLayout(vbox)

        if self.singleImage:
            self.rows = 1
            self.columns = 1
        else:
            self.rows = 4
            self.columns = 4
            self.row_box.setValue(self.rows)
            self.column_box.setValue(self.columns)

        self.random_box.setCheckState(False)

