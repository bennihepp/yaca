# -*- coding: utf-8 -*-

"""
main_gui.py -- Runs the GUI.
"""

# This software is distributed under the FreeBSD License.
# See the accompanying file LICENSE for details.
# 
# Copyright 2011 Benjamin Hepp

import sys
import os

TRY_OPENGL = False

project_file = None
start_pipeline = False


#TRY_OPENGL = False

#simple_ui = False
#project_file = None
#start_pipeline = False
#img_symlink_dir = False

#skip_next = 0


#def print_help():

    #sys.stderr.write( """Usage: python %s [options]
#Possible options:
  #--simple-ui                   Use a simplified user interface (e.g. for presentation)
  #--full-ui                     Use the full user interface (default)
  #--opengl                      Use OpenGL for rendering
  #--no-opengl                   Don't use OpenGL for rendering (default)
  #--project-file <filename>     Load specified pipeline file
  #--run-filtering               Run quality control and pre-filtering after loading
                                #of the project file
  #--img-symlinks <path>         Create symlinks of the images within <path>,
                                #the filename will be the image-ID.
#""" % sys.argv[ 0 ] )


#if len( sys.argv ) > 1:
    #for i in xrange( 1, len( sys.argv ) ):

        #arg = sys.argv[ i ]

        #if skip_next > 0:
            #skip_next -= 1
            #continue

        #if arg == '--simple-ui':
            #simple_ui = True
        #elif arg == '--full-ui':
            #simple_ui = False
        #elif arg == '--opengl':
            #TRY_OPENGL = True
        #elif arg == '--no-opengl':
            #TRY_OPENGL = False
        #elif arg == '--run-filtering':
            #start_pipeline = True
        #elif arg == '--img-symlinks':
            #img_symlink_dir= sys.argv[ i+1 ]
            #skip_next = 1
        #elif arg == '--project-file':
            #project_file = sys.argv[ i+1 ]
            #skip_next = 1
        #elif arg == '--help':
            #print_help()
            #sys.exit( 0 )
        #else:
            #sys.stderr.write( 'Unknown option: %s\n' % arg )
            #print_help()
            #sys.exit( -1 )


from PyQt4.QtCore import *
from PyQt4.QtGui import *

from gui.main_window import MainWindow


#g = gui_window.GUI(pdc, objFeatures, mahalPoints, mahalFeatureNames, channelDescription, partition, sorting, inverse_sorting, clusters, cluster_count, sys.argv)

def run(args):

    #if img_symlink_dir:

        #print 'creating image symlinks...'

        #if not os.path.isdir( img_symlink_dir ):
            #os.mkdir( img_symlink_dir )

        #for i in xrange( len( pdc.images ) ):

            #img = pdc.images[ i ]
            #imgId = img.index
            #imgFiles = img.imageFiles

            #for name,path in imgFiles:
                #symlink_name = '%04d_%s.tif' % ( imgId, name )
                #symlink_path = os.path.join( img_symlink_dir, symlink_name )
                #os.symlink( path, symlink_path )

        #print 'finished creating image symlinks'

    TRY_OPENGL = args.opengl
    project_file = args.project_file
    start_pipeline = args.run_filtering

    app = QApplication( sys.argv )

    #mainwindow = MainWindow( simple_ui )
    mainwindow = MainWindow()
    mainwindow.show()

    if project_file:
        mainwindow.load_project_file( project_file )
        if start_pipeline:
            mainwindow.on_start_cancel()

    app.exec_()
    mainwindow.close()

    """importer = mainwindow.importer
    pdc = importer.get_pdc()


    for k,v in pdc.images[0].properties.iteritems():
        print '%s=%s' % (k,v)

    keys = pdc.imgFeatureIds.keys()
    keys.sort()
    for k in keys:
        id = pdc.imgFeatureIds[ k ]
        v = pdc.imgFeatures[ 0 , id ]
        print '%s=%f' % (k,v)

    return pdc"""
