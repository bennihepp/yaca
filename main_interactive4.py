import sys, os
import time
import numpy

from src.core import pipeline
from src.core import importer
from src.core import analyse
from src.core import distance
from src.core import headless_cluster_configuration
from src.core import headless_channel_configuration

import src.main_gui
from src.gui.gallery_window import GalleryWindow
from src.gui.gui_utils import CellPixmapFactory,CellFeatureTextFactory

from src.core import parameter_utils as utils

#import hcluster as hc




skip_next = 0


def print_help():

    sys.stderr.write( """Usage: python %s [options]
Necessary options:
  --project-file <filename>     Load specified pipeline file
""" % sys.argv[ 0 ] )


if len( sys.argv ) > 1:
    for i in xrange( 1, len( sys.argv ) ):

        arg = sys.argv[ i ]

        if skip_next > 0:
            skip_next -= 1
            continue

        if arg == '--project-file':
            project_file = sys.argv[ i+1 ]
            skip_next = 1
        elif arg == '--help':
            print_help()
            sys.exit( 0 )
        else:
            sys.stderr.write( 'Unknown option: %s\n' % arg )
            print_help()
            sys.exit( -1 )


#from PyQt4.QtCore import *
#from PyQt4.QtGui import *

#from gui.main_window import MainWindow


#g = gui_window.GUI(pdc, objFeatures, mahalPoints, mahalFeatureNames, channelDescription, partition, sorting, inverse_sorting, clusters, cluster_count, sys.argv)

yaca_importer = importer.Importer()

headlessClusterConfiguration = headless_cluster_configuration.HeadlessClusterConfiguration()
headlessChannelConfiguration = headless_channel_configuration.HeadlessChannelConfiguration()

print 'Loading project file...'
utils.load_module_configuration( project_file )
print 'Project file loaded'


modules = utils.list_modules()
for module in modules:

    if not utils.all_parameters_set( module ):

        print 'Not all required parameters for module %s have been set' % module
        print 'Unable to start pipeline'
        sys.exit( 1 )

    elif not utils.all_requirements_met( module ):

        print 'Not all requirements for module %s have been fulfilled' % module
        print 'Unable to start pipeline'
        sys.exit( 1 )

pdc = yaca_importer.get_pdc()
clusterConfiguration = headlessClusterConfiguration.clusterConfiguration

pl = pipeline.Pipeline( pdc, clusterConfiguration )
pl.run_quality_control()

pl.run_pre_filtering( analyse.FILTER_MODE_xMEDIAN )


import mixture
import matplotlib
from matplotlib import pyplot as plt


treatmentId = pdc.treatmentByName[ 'S24A[40]' ]

mask1 = pl.mask_and(
    pl.get_control_treatment_cell_mask(),
    pl.get_valid_cell_mask(),
    pl.get_non_control_cell_mask()
)
mask2 = pl.mask_and(
    pl.get_treatment_cell_mask( treatmentId ),
    pl.get_valid_cell_mask(),
    pl.get_non_control_cell_mask()
)
mask = pl.mask_or( mask1, mask2 )

features = pdc.objFeatures

print 'Feature set: %s' % str( clusterConfiguration[ 1 ][ 0 ] )
featureIds = []
featureNames = clusterConfiguration[ 1 ][ 1 ]
for featureName in featureNames:
    featureIds.append( pdc.objFeatureIds[ featureName ] )
featureIds = numpy.array( featureIds )

features = features[ : , featureIds ]

newFeatureIds = featureIds


from src.core.batch import utils as batch_utils

NUM_OF_FEATURES = 8
BOOTSTRAP_COUNT_MULTIPLIER = 0.1
BOOTSTRAP_COUNT_MAX = 100
BOOTSTRAP_SAMPLE_SIZE_RATIO = 0.1
#RESAMPLE_MAX = 3

norm_features = pdc.objFeatures[ : , newFeatureIds ]

ctrl_mask = pl.mask_and( pl.get_control_treatment_cell_mask(), pl.get_valid_cell_mask() )
obs = pl.pdc.objFeatures[ ctrl_mask ][ : , newFeatureIds ]
mask3 = pl.mask_and( pl.get_non_control_treatment_cell_mask(), pl.get_valid_cell_mask() )
bootstrap_sample_size = round( BOOTSTRAP_SAMPLE_SIZE_RATIO * obs.shape[0] )
obs1 = numpy.empty( ( bootstrap_sample_size, newFeatureIds.shape[0] ) )
#self.pdc.objFeatures[ mask1 ][ : , newFeatureIds ]
#edf1 = batch_utils.compute_edf( obs1 )
obs3 = pl.pdc.objFeatures[ mask3 ][ : , newFeatureIds ]
edf3 = batch_utils.compute_edf( obs3 )
obs2 = numpy.empty( obs1.shape )
bootstrap_count = int( BOOTSTRAP_COUNT_MULTIPLIER * obs.shape[0] + 0.5 )
bootstrap_count = numpy.min( [ bootstrap_count, BOOTSTRAP_COUNT_MAX ] )
dist = numpy.empty( ( bootstrap_count + 1, norm_features.shape[1] ) )
bootstrap_dist = numpy.empty( ( bootstrap_count, norm_features.shape[1] ) )
print 'bootstrapping ks statistics (%d)...' % ( bootstrap_count )
def plot_edf(edf, *args, **kwargs):
    print 'edf.shape:', edf.shape
    unique_edf = numpy.unique( edf )
    print 'unique_edf.shape:', unique_edf.shape
    X = []
    Y = []
    #y = numpy.arange( unique_edf.shape[0] + 2 * ( edf.shape[0] - unique_edf.shape[0] ) ) + 1
    #x = numpy.empty( y.shape )
    x = edf[0]
    y = 0
    X.append( x )
    Y.append( y )
    old_v = edf[0]
    for v in edf:
        if old_v == v:
            y += 1
            X.append( x )
            Y.append( y )
        else:
            x = v
            X.append( x )
            Y.append( y )
            y += 1
            X.append( x )
            Y.append( y )
    X = numpy.array( X )
    Y = numpy.array( Y )
    plt.plot(X,Y,*args,**kwargs)
for i in xrange( bootstrap_count ):
    sys.stdout.write('\riteration %d...' % ( i+1 ))
    sys.stdout.flush()
    resample_ids = numpy.random.randint( 0, obs.shape[0], 2*obs1.shape[0] )
    obs1 = obs[ resample_ids[:obs1.shape[0]] ]
    obs2 = obs[ resample_ids[obs1.shape[0]:] ]
    edf1 = batch_utils.compute_edf( obs1 )
    edf2 = batch_utils.compute_edf( obs2 )
    for k in xrange( norm_features.shape[1] ):
        support1 = edf1[ k ]
        support2 = edf2[ k ]
        support3 = edf3[ k ]
        bootstrap_dist[ i, k ] = batch_utils.compute_edf_distance( support1, support2 )
        dist1 = batch_utils.compute_edf_distance( support1, support3 )
        dist2 = batch_utils.compute_edf_distance( support2, support3 )
        dist[ i, k ] = max( dist1, dist2 )
        #print 'bootstrap_dist:', bootstrap_dist[i,k]
        #print 'dist:', dist[i,k]
        #plot_edf(edf1[k], color='red')
        #plot_edf(edf2[k], color='green')
        #plot_edf(edf3[k], color='blue')
        #plt.show()
print '\rfinished bootstrapping'
edf = batch_utils.compute_edf( obs )
for k in xrange( norm_features.shape[1] ):
    support = edf[ k ]
    support3 = edf3[ k ]
    dist[ bootstrap_count, k ] = batch_utils.compute_edf_distance( support, support3 )

print 'obs.shape:', obs.shape
print 'obs1.shape:', obs1.shape
print 'obs3.shape:', obs3.shape

#plt.plot( numpy.arange( edf1.shape[1] ), edf1[1], color='red' )
#plt.plot( numpy.arange( edf2.shape[1] ), edf2[1], color='green' )
#plt.plot( numpy.arange( edf3.shape[1] ), edf3[1], color='blue' )
#plt.show()

max_dist = numpy.max( dist[:-1], axis=0 )
mean_dist = numpy.mean( dist[:-1], axis=0 )
median_dist = numpy.median( dist[:-1], axis=0 )
stddev_dist = numpy.std( dist[:-1], axis=0 )

max_bootstrap_dist = numpy.max( bootstrap_dist, axis=0 )
mean_bootstrap_dist = numpy.mean( bootstrap_dist, axis=0 )
median_bootstrap_dist = numpy.median( bootstrap_dist, axis=0 )
stddev_bootstrap_dist = numpy.std( bootstrap_dist, axis=0 )

def select_best_features(feature_quality, num_of_features=3):
    best_feature_mask = numpy.ones( ( feature_quality.shape[0], ), dtype=bool )
    if num_of_features > 0:
        num_of_features = min( num_of_features, feature_quality.shape[0] )
        sorted_feature_indices = numpy.argsort( feature_quality )
        best_feature_mask[ sorted_feature_indices[ : -num_of_features ] ] = False
    return best_feature_mask

print 'selecting %d best features...' % NUM_OF_FEATURES

feature_quality = dist[-1] - mean_bootstrap_dist
best_feature_mask = select_best_features( feature_quality, NUM_OF_FEATURES )
norm_features = norm_features[ : , best_feature_mask ]
newFeatureIds = newFeatureIds[ best_feature_mask ]

print 'best informative features:'
print newFeatureIds
for feature_id in newFeatureIds:
    fname = None
    for fn,fid in pl.pdc.objFeatureIds.iteritems():
        if fid == feature_id:
            fname = fn
            break
    print 'feature (%d): %s' % ( feature_id, fname )

xlabels = []
for fid in featureIds:
    xlabels.append( str( fid ) )
left = 4 * numpy.arange( obs.shape[1] )
plt.bar( left, mean_bootstrap_dist, yerr=stddev_bootstrap_dist, facecolor='red', align='center' )
left = left + 1.0
plt.bar( left, mean_dist, yerr=stddev_dist, facecolor='green', align='center' )
left = left + 1.0
plt.bar( left, dist[-1], facecolor='blue', align='center' )
left = left + 1.0
plt.bar( left, feature_quality, facecolor='yellow', align='center' )
t = 4 * numpy.arange( obs.shape[1] ) + 0.5
plt.axes().set_xticks( t )
plt.axes().set_xticklabels( xlabels )
plt.show()


#ids = numpy.arange( valid_mask.shape[0] )[ valid_mask ][ best_feature_mask ]
#valid_mask = numpy.zeros( valid_mask.shape, dtype=bool )
#valid_mask[ ids ] = True
#ids = fids[ valid_mask ]
valid_mask2 = numpy.zeros( valid_mask.shape, dtype=bool )
valid_mask2[ valid_mask ] = best_feature_mask
valid_mask = valid_mask2


xlabels = []
for fid in featureIds:
    xlabels.append( str( fid ) )
left = 2 * numpy.arange( obs1.shape[1] )
plt.bar( left, median_bootstrap_dist, facecolor='red', align='center' )
left = left + 1.0
plt.bar( left, median_dist, facecolor='green', align='center' )
t = 2 * numpy.arange( obs1.shape[1] ) + 0.5
plt.axes().set_xticks( t )
plt.axes().set_xticklabels( xlabels )
plt.show()

best_features = features[ : , best_feature_mask ]
#features = features[ : 1000 ]
#features = features[ : , : 3 ]
mu1 = numpy.mean( best_features[ mask1 ], axis=0 )
mu2 = numpy.mean( best_features[ mask2 ], axis=0 )
sigma1 = numpy.cov( best_features[ mask1 ], rowvar=0 )
sigma2 = numpy.cov( best_features[ mask2 ], rowvar=0 )
p1 = numpy.sum( mask1 ) / float( numpy.sum( mask ) )
p2 = 1.0 - p1
#best_features = best_features[ mask ]

