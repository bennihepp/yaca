import sys, os
import time
import numpy

from src.core import pipeline
from src.core import importer
from src.core import analyse
from src.core import headless_cluster_configuration
from src.core import headless_channel_configuration

import src.main_gui
from src.gui.gallery_window import GalleryWindow
from src.gui.gui_utils import CellPixmapFactory,CellFeatureTextFactory

from src.core import parameter_utils as utils

import hcluster as hc



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








norm_features = features
valid_mask = numpy.ones( ( features.shape[1], ), dtype=bool )
newFeatureIds = featureIds

#norm_features, valid_mask, newFeatureIds = pl.compute_normalized_features( mask, featureset_index, self.get_control_treatment_cell_mask() )
#print 'control_treatment_cell_maskk:', self.get_control_treatment_cell_mask().shape, self.get_control_treatment_cell_mask().dtype
#print 'nonControlCellMask:', self.nonControlCellMask.shape, self.nonControlCellMask.dtype

best_feature_mask = numpy.ones( ( norm_features.shape[1], ), dtype=bool )

do_feature_selection = True
if do_feature_selection:

    from src.core.batch import utils as batch_utils

    NUM_OF_FEATURES = 5
    BOOTSTRAP_COUNT_MULTIPLIER = 0.1
    BOOTSTRAP_COUNT_MAX = 500
    #RESAMPLE_MAX = 3

    mask1 = pl.mask_and( pl.get_control_treatment_cell_mask(), pl.get_valid_cell_mask() )
    mask2 = pl.mask_and( pl.get_non_control_treatment_cell_mask(), pl.get_valid_cell_mask() )
    obs1 = pl.pdc.objFeatures[ mask1 ][ : , newFeatureIds ]
    edf1 = batch_utils.compute_edf( obs1 )
    obs3 = pl.pdc.objFeatures[ mask2 ][ : , newFeatureIds ]
    edf3 = batch_utils.compute_edf( obs3 )
    obs2 = numpy.empty( obs1.shape )
    bootstrap_count = int( BOOTSTRAP_COUNT_MULTIPLIER * obs1.shape[0] + 0.5 )
    bootstrap_count = numpy.min( [ bootstrap_count, BOOTSTRAP_COUNT_MAX ] )
    dist = numpy.empty( ( bootstrap_count + 1, norm_features.shape[1] ) )
    bootstrap_dist = numpy.empty( ( bootstrap_count, norm_features.shape[1] ) )
    print 'bootstrapping ks statistics (%d)...' % ( bootstrap_count )
    for i in xrange( bootstrap_count ):
        resample_ids = numpy.random.randint( 0, obs1.shape[0], obs1.shape[0] )
        obs2 = obs1[ resample_ids ]
        edf2 = batch_utils.compute_edf( obs2 )
        for k in xrange( norm_features.shape[1] ):
            support1 = edf1[ k ]
            support2 = edf2[ k ]
            support3 = edf3[ k ]
            bootstrap_dist[ i, k ] = batch_utils.compute_edf_distance( support1, support2 )
            dist[ i, k ] = batch_utils.compute_edf_distance( support2, support3 )
    for k in xrange( norm_features.shape[1] ):
        support1 = edf1[ k ]
        support3 = edf3[ k ]
        dist[ bootstrap_count, k ] = batch_utils.compute_edf_distance( support1, support3 )

    max_dist = numpy.max( dist, axis=0 )
    mean_dist = numpy.mean( dist, axis=0 )
    median_dist = numpy.median( dist, axis=0 )

    max_bootstrap_dist = numpy.max( bootstrap_dist, axis=0 )
    mean_bootstrap_dist = numpy.mean( bootstrap_dist, axis=0 )
    median_bootstrap_dist = numpy.median( bootstrap_dist, axis=0 )

    def select_best_features(feature_quality, num_of_features=3):
        best_feature_mask = numpy.ones( ( feature_quality.shape[0], ), dtype=bool )
        if num_of_features > 0:
            num_of_features = min( num_of_features, feature_quality.shape[0] )
            sorted_feature_indices = numpy.argsort( feature_quality )
            best_feature_mask[ sorted_feature_indices[ : -num_of_features ] ] = False
        return best_feature_mask

    print 'selecting %d best features...' % NUM_OF_FEATURES

    feature_quality = median_dist - median_bootstrap_dist
    best_feature_mask = select_best_features( feature_quality, NUM_OF_FEATURES )
    norm_features = norm_features[ : , best_feature_mask ]
    newFeatureIds = newFeatureIds[ best_feature_mask ]
    #ids = numpy.arange( valid_mask.shape[0] )[ valid_mask ][ best_feature_mask ]
    #valid_mask = numpy.zeros( valid_mask.shape, dtype=bool )
    #valid_mask[ ids ] = True
    #ids = fids[ valid_mask ]
    valid_mask2 = numpy.zeros( valid_mask.shape, dtype=bool )
    valid_mask2[ valid_mask ] = best_feature_mask
    valid_mask = valid_mask2

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
left = 2 * numpy.arange( obs1.shape[1] )
plt.bar( left, median_bootstrap_dist, facecolor='red', align='center' )
left = left + 1.0
plt.bar( left, median_dist, facecolor='green', align='center' )
t = 2 * numpy.arange( obs1.shape[1] ) + 0.5
plt.axes().set_xticks( t )
plt.axes().set_xticklabels( xlabels )
#plt.show()

norm_features = norm_features[ mask ]

from scipy import stats

from scipy import linalg, special
from numpy import atleast_2d, reshape, zeros, newaxis, dot, exp, pi, sqrt, \
     ravel, power, atleast_1d, squeeze, sum, transpose
from numpy.random import randint, multivariate_normal

class custom_gaussian_kde(stats.kde.gaussian_kde):

    def __init__(self, dataset):
        stats.kde.gaussian_kde.__init__(self, dataset)

    def evaluate(self, points, user_function=None):
        """Evaluate the estimated pdf on a set of points.

        Parameters
        ----------
        points : (# of dimensions, # of points)-array
            Alternatively, a (# of dimensions,) vector can be passed in and
            treated as a single point.

        Returns
        -------
        values : (# of points,)-array
            The values at each point.

        Raises
        ------
        ValueError if the dimensionality of the input points is different than
        the dimensionality of the KDE.
        """

        points = atleast_2d(points).astype(self.dataset.dtype)

        d, m = points.shape
        if d != self.d:
            if d == 1 and m == self.d:
                # points was passed in as a row vector
                points = reshape(points, (self.d, 1))
                m = 1
            else:
                msg = "points have dimension %s, dataset has dimension %s" % (d,
                    self.d)
                raise ValueError(msg)

        #result = zeros((d,m), points.dtype)
        result = None

        if m >= self.n:
            # there are more points than data, so loop over data
            for i in range(self.n):
                diff = self.dataset[:,i,newaxis] - points
                tdiff = dot(self.inv_cov, diff)
                energy = sum(diff*tdiff,axis=0)/2.0
                if user_function != None:
                    user_weight = user_function( self.dataset[:,i].T )
                    temp_result = dot( user_weight[:,newaxis], exp(-energy)[newaxis,:] )
                else:
                    temp_result = exp(-energy)
                if result == None:
                    if user_function != None:
                        result = zeros((user_weight.shape[0],m), points.dtype)
                    else:
                        result = zeros((m,), points.dtype)
                result += temp_result
        else:
            # loop over points
            for i in range(m):
                diff = self.dataset - points[:,i,newaxis]
                tdiff = dot(self.inv_cov, diff)
                energy = sum(diff*tdiff,axis=0)/2.0
                if user_function != None:
                    user_weight = user_function( self.dataset.T )
                    temp_result = sum( exp(-energy)[:,newaxis] * user_weight, axis=0 )
                else:
                    temp_result = sum( exp(-energy), axis=0 )
                if result == None:
                    if user_function != None:
                        result = zeros((user_weight.shape[1],m), points.dtype)
                    else:
                        result = zeros((m,), points.dtype)
                if user_function != None:
                    result[:,i] = temp_result
                else:
                    result[i] = temp_result

        result /= self._norm_factor

        return result

    __call__ = evaluate




kernel = custom_gaussian_kde( norm_features.T )
Z = kernel( norm_features.T )

plt.scatter( numpy.arange( Z.shape[0] ), Z )
#plt.show()

epsilon = 0.01
k = 2


def hill_climbing(x, pdf_kernel, k=2, epsilon=0.01, max_iterations=1000):
    x = numpy.array( x )
    x_new = numpy.empty( x.shape )
    mask = numpy.ones( ( x.shape[0], ), dtype=bool )
    density = pdf_kernel( x[ mask ].T )
    c = numpy.zeros( ( x.shape[0], ), dtype=int )
    s = numpy.zeros( ( x.shape[0], k ) )
    max_iterations = max( k, max_iterations )
    for i in xrange( max_iterations ):
        print 'iteration:', (i+1), 'points left:', numpy.sum( mask )
        x_new[ mask ] = pdf_kernel( x[ mask ].T, lambda y: y ).T / density[:,newaxis]
        new_density = pdf_kernel( x_new[ mask ].T ).T
        density_diff = numpy.abs( density - new_density ) / new_density
        x_diff = numpy.sqrt( numpy.sum( ( x_new[ mask ] - x[ mask ] )**2, axis=1 ) )
        s[ mask, ( i % k ) ] = x_diff
        x[ mask ] = x_new[ mask ]
        c[ mask ] += 1
        new_mask = numpy.logical_or( density_diff > epsilon, c[ mask ] < k )
        new_density = new_density[ new_mask ]
        density = new_density
        mask[ mask ] = new_mask
        if numpy.sum( mask ) == 0:
            break
    print 'stopping after %d iterations' % (i+1)
    s = numpy.sum( s, axis=1 )
    return x,s

def find_partitioning(x, s):
    _arange = numpy.arange( x.shape[0] )
    mask = numpy.zeros( (x.shape[0],), dtype=bool )
    partition = -numpy.ones( (x.shape[0],), dtype=int )
    for i in range( x.shape[0] ):
        p = x[i]
        dist = numpy.sqrt( numpy.sum( ( p[newaxis,:] - x[:i] )**2, axis=1 ) )
        m = dist <= (s[:i] + s[i])
        #m[i] = False
        if numpy.any( m ):
            cids = numpy.unique( partition[:i][ m ] )
            if cids.shape[0] > 1:
                # multiple corresponding clusters exist
                mask[ i ] = True
                mask[ _arange[:i][ m ] ] = True
                #ambigous_clusters.extend( cids )
                partition[ i ] = cids[ 0 ]
            else:
                # single corresponding cluster already exists
                partition[ i ] = cids[ 0 ]
        else:
            # new cluster has to be created
            partition[ i ] = numpy.max( partition ) + 1
    return partition, mask

y = norm_features.copy()
s = numpy.empty( (norm_features.shape[0],) )
partition = numpy.zeros( (norm_features.shape[0],), dtype=int )
mask = numpy.ones( (norm_features.shape[0],), dtype=bool )
while numpy.sum( mask ) > 0:
    #for cluster in ambigous_clusters:
    #    mask = numpy.logical_or( mask, partition == cluster )
    print 'running hill climbing for %d cells' % numpy.sum( mask )
    y[mask],s[mask] = hill_climbing( y[mask], kernel, k, epsilon )
    partition, mask = find_partitioning( y, s )

#partition, mask = find_partitioning( y, s )
