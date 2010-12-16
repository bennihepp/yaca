import sys
import os

import sys

TRY_OPENGL = True

if len( sys.argv ) > 1:
    for arg in sys.argv[1:]:
        if arg == '-opengl':
            TRY_OPENGL = False


from PyQt4.QtCore import *
from PyQt4.QtGui import *

from gui.main_window import MainWindow


#g = gui_window.GUI(adc, objFeatures, mahalPoints, mahalFeatureNames, channelDescription, partition, sorting, inverse_sorting, clusters, cluster_count, sys.argv)

def run():

    app = QApplication( sys.argv )

    mainwindow = MainWindow()
    mainwindow.show()

    app.exec_()
    mainwindow.close()

    importer = mainwindow.importer
    adc = importer.get_adc()

    """for k,v in adc.images[0].properties.iteritems():
        print '%s=%s' % (k,v)

    keys = adc.imgFeatureIds.keys()
    keys.sort()
    for k in keys:
        id = adc.imgFeatureIds[ k ]
        v = adc.imgFeatures[ 0 , id ]
        print '%s=%f' % (k,v)

    return adc"""

