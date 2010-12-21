import sys
import os

import sys

TRY_OPENGL = True

project_file = None
start_pipeline = False
img_symlink_dir = False

skip_next = 0

if len( sys.argv ) > 1:
    for i in xrange( 1, len( sys.argv ) ):

        arg = sys.argv[ i ]

        if skip_next > 0:
            skip_next -= 1
            continue

        if arg == '-opengl':
            TRY_OPENGL = False
        elif arg == '-start':
            start_pipeline = True
        elif arg == '-img-symlinks':
            img_symlink_dir= sys.argv[ i+1 ]
            skip_next = 1
        elif arg == '-project-file':
            project_file = sys.argv[ i+1 ]
            skip_next = 1


from PyQt4.QtCore import *
from PyQt4.QtGui import *

from gui.main_window import MainWindow


#g = gui_window.GUI(adc, objFeatures, mahalPoints, mahalFeatureNames, channelDescription, partition, sorting, inverse_sorting, clusters, cluster_count, sys.argv)

def run():

    app = QApplication( sys.argv )

    mainwindow = MainWindow()
    mainwindow.show()

    if project_file:
        mainwindow.load_project_file( project_file )
        if start_pipeline:
            mainwindow.on_start_cancel()

    app.exec_()
    mainwindow.close()

    importer = mainwindow.importer
    adc = importer.get_adc()

    if img_symlink_dir:

        if not os.path.isdir( img_symlink_dir ):
            os.mkdir( img_symlink_dir )

        for i in xrange( len( adc.images ) ):

            img = adc.images[ i ]
            imgId = img.rowId
            imgFiles = img.imageFiles

            for name,path in imgFiles:
                symlink_name = '%04d_%s.tif' % ( imgId, name )
                symlink_path = os.path.join( img_symlink_dir, symlink_name )
                os.symlink( path, symlink_path )


    """for k,v in adc.images[0].properties.iteritems():
        print '%s=%s' % (k,v)

    keys = adc.imgFeatureIds.keys()
    keys.sort()
    for k in keys:
        id = adc.imgFeatureIds[ k ]
        v = adc.imgFeatures[ 0 , id ]
        print '%s=%f' % (k,v)

    return adc"""

