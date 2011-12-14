# -*- coding: utf-8 -*-

"""
pipeline.py -- Computational pipeline.

This module contains the Pipeline class. This class defines the whole
computational workflow of the application.
The command-line interface or the graphical user-interface only need to know
about this class. All the steps like quality-control, pre-filtering and
clustering are represented as methods in the Pipeline class.
"""

# This software is distributed under the FreeBSD License.
# See the accompanying file LICENSE for details.
# 
# Copyright 2011 Benjamin Hepp

import sys
import os
import time

from thread_utils import Thread,SIGNAL

import numpy

try:
    import hcluster
    has_hcluster = True
except:
    has_hcluster = False

import analyse, distance, quality_control, cluster, cluster_profiles, grouping


# define necessary parameters for this module
# (see parameter_utils.py for details)
#
import parameter_utils as utils
#
__dict__ = sys.modules[ __name__ ].__dict__
#
utils.register_module( __name__, 'Image analysis pipeline', __dict__ )
#
import importer
utils.add_required_state( __name__, importer.__name__, 'imported' )
#
utils.register_parameter( __name__, 'PCA_components', utils.PARAM_INT, 'Number of PCAs to extract', 3, 1, 100 )
#
utils.register_parameter( __name__, 'number_of_clusters', utils.PARAM_INT, 'Number of clusters (Default is number of treatment)', None, 1, None, True )
#
utils.register_parameter( __name__, 'clustering_swap_threshold', utils.PARAM_INT, 'Swap threshold for the clustering', 1, 1, None )
#
utils.register_parameter( __name__, 'fermi_dirac_sim_sharpness', utils.PARAM_FLOAT, 'Sharpness for the fermi-dirac-similarity', 2.0, 0.0, None )
#
utils.register_parameter( __name__, 'minkowski_p', utils.PARAM_INT, 'Parameter p for the Minkowski metric', 2, 1, None )
#
utils.register_parameter( __name__, 'num_of_reference_datasets', utils.PARAM_INT, 'Number of reference datasets to sample for GAP statistics', 5, 1, None )
#
utils.register_parameter( __name__, 'discard_cluster_threshold', utils.PARAM_INT, 'Minimum size of clusters which will not be discarded', 1, 0, 10000 )
#
utils.register_parameter( __name__, 'dissolve_cluster_threshold', utils.PARAM_INT, 'Minimum size of clusters which will not be dissolved', 200, 0, 10000 )
#
utils.register_parameter( __name__, 'merge_cluster_factor', utils.PARAM_FLOAT, 'Weighting of the standard deviations when merging clusters', 1.0, -100.0, 100.0 )
#
utils.register_parameter( __name__, 'merge_cluster_offset', utils.PARAM_FLOAT, 'Offset for the minimum distance when merging clusters', 0.0, -100.0, 100.0 )
#
utils.register_parameter( __name__, 'reject_treatments', utils.PARAM_TREATMENTS, 'Treatments not to be used for clustering' )


class Pipeline(Thread):
    """The Pipeline class represents the computational workflow. It represents all the steps
    like  quality-control, pre-filtering and clustering as methods

    Public methods:
    Public members:
    """

    __pyqtSignals__ = ('updateProgress',)


    PIPELINE_STATE_INIT = 'init'
    PIPELINE_STATE_QUALITY_CONTROL = 'quality_control'
    PIPELINE_STATE_PRE_FILTERING = 'pre_filtering'
    PIPELINE_STATE_CLUSTERING = 'clustering'

    PIPELINE_STATE_ATTRIBUTES = [
        'clusterConfiguration',
        'validImageMask',
        'validCellMask',
        'controlCellMask',
        'nonControlCellMask',
        'featureIds',
        'featureNames',
        'nonControlFeatures',
        'nonControlMahalDist',
        'nonControlClusterFeatures',
        'clusters',
        'partition',
        'nonControlSilhouette',
        'nonControlWeights',
        'intraClusterDistances',
        'interClusterDistances',
        'nonControlFeatureIds',
        'nonControlFeatureNames',
        'nonControlClusterProfiles',
        'clusterContainer'
    ]


    def __init__(self, pdc, clusterConfiguration):
        """Constructor for Pipeline
    
        Input parameters:
            - pdc: YACA data container
            - clusterConfiguration: contains a list of super-clusters.
              each super-cluster is represented as a list of the features to use
        """

        # initialize the base class
        Thread.__init__( self )

        # keep the pdc and the clusterConfiguration for later use
        self.pdc = pdc
        self.clusterConfiguration = clusterConfiguration

        #TODO
        self.pca = None

        # initialize some public members
        # the mask of images and cells that passed quality control
        self.validImageMask = None
        self.validCellMask = None
        # the mask of control-like and not-control-like cells (that passed quality control)
        self.controlCellMask = None
        self.nonControlCellMask = None
        #self.clusterCellMask = None
        # ?? TODO ??
        #self.featureIds = None
        #self.featureNames = None
        # the features of the not-control-like cells
        #self.nonControlFeatures = None
        # the features of the not-control-like cells used for clustering
        #self.nonControlClusterFeatures = None
        # the clusters of the non-control-like cells that were found
        self.clusters = None
        # the partition of the non-control-like cells that was found
        self.partition = None
        # ?? TODO ?? the silhouette of the non-control-like cells that was found
        #self.nonControlSilhouette = None
        #
        self.validTreatments = None
        self.validTreatmentIds = None

        # this keeps the callback method for the progress of a step within the pipeline
        self.progressCallback = None

        # this keeps the result of a step within the pipeline (if it was run in a thread)
        self.result = None

        # this keeps the current state of the pipeline
        self.__state = self.PIPELINE_STATE_INIT

    def __del__(self):
        """Destructor for Pipeline
        """

        # wait until the thread has finished
        self.wait()

    def load_state(self, file):

        filename = None
        if type( file ) == str:
            filename = file
            file = open( file, 'r' )

        try:

            import cPickle

            p = cPickle.Unpickler( file )
            dict = p.load()

            self.__state = dict[ '__state' ]

            for attr in self.PIPELINE_STATE_ATTRIBUTES:
                self.__setattr__( attr, dict[ attr ] )

        finally:

            if filename != None:
                file.close()

    def save_state(self, file):

        dict = {}

        dict[ '__state' ] = self.__state

        for attr in self.PIPELINE_STATE_ATTRIBUTES:
            dict[ attr ] = self.__getattribute__( attr )

        filename = None
        if type( file ) == str:
            filename = file
            file = open( file, 'w' )

        try:

            import cPickle

            p = cPickle.Pickler( file )
            p.dump( dict )

        finally:

            if filename != None:
                file.close()

    def load_clusters(self, file,  exp_factor=10.0):

        filename = None
        if type( file ) == str:
            filename = file
            file = open( file, 'r' )

        try:

            import cPickle

            p = cPickle.Unpickler( file )
            dict = p.load()

            #clusters = dict[ 'clusters' ]
            #featureNames = dict[ 'featureNames' ]

            clusterContainer = dict[ 'clusterContainer' ]

            print 'keys:'
            print '\n'.join( clusterContainer.keys() )

            clusters = clusterContainer[ 'clusters' ]    
            featureNames = clusterContainer[ 'featureNames' ]
            Z = clusterContainer[ 'Z' ]
            tree = Z
            method = clusterContainer[ 'method' ]
            dump_cluster_index = -1
            if 'dump_cluster_index' in clusterContainer:
                dump_cluster_index = clusterContainer[ 'dump_cluster_index' ]

            # determine the IDs of the features to be used
            featureIds = []
            for featureName in featureNames:
                featureIds.append( self.pdc.objFeatureIds[ featureName ] )

            id_mapping = numpy.array( featureIds )
    
            # extract the features for the clustering
            features = self.pdc.objFeatures[ self.nonControlCellMask ][ : , featureIds ]
    
            norm_features, valid_feature_mask, newFeatureIds = self.__compute_normalized_features( self.nonControlCellMask, featureNames, self.get_control_treatment_cell_mask() )
            newFeatureIds = clusterContainer[ 'newFeatureIds' ]
            valid_feature_mask = clusterContainer[ 'valid_feature_mask' ]
            best_feature_mask = clusterContainer[ 'best_feature_mask' ]

            norm_features = norm_features[ : , best_feature_mask ]

            #print 'norm_features[ : 20, 5 ]:'
            #print norm_features[ : 20, 5 ]
            #print 'norm_features[ : 20, 10 ]:'
            #print norm_features[ : 20, 10 ]
            #print 'norm_features[ : 20, 15 ]:'
            #print norm_features[ : 20, 15 ]

            if method != 'kd-tree':

                partition = self.__partition_along_clusters( norm_features, clusterContainer )
                #partition = self.__partition_along_clusters( norm_features, clusters )

            else:

                partition = cluster.partition_along_kd_tree( tree, norm_features )

            """if 'nonControlClusterProfiles' in clusterContainer:
                clusterProfiles = clusterContainer[ 'nonControlClusterProfiles' ]
                print 'loading saved cluster profiles...'
            else:
                clusterProfiles = self.__compute_cluster_profiles( self.nonControlCellMask, norm_features, clusters, partition, exp_factor )
                print 'computing new cluster profiles...'"""

            print 'computing cluster profiles for treatment masks...'
            masks = []
            for tr in self.pdc.treatments:
                mask = self.get_treatment_cell_mask( tr.index )[ self.get_non_control_cell_mask() ]
                masks.append( mask )
            clusterProfiles = cluster_profiles.compute_cluster_profiles( masks, norm_features, clusters, -1, 2.0 )

    
            mask = partition >= 0
            clusterContainer = {
                'points' : norm_features[ mask ].copy(),
                'partition' : partition[ mask ].copy(),
                'clusters' : clusters.copy(),
                'dump_cluster_index' : dump_cluster_index,
                'method' : method,
                'mask' : mask.copy(),
                'featureNames' : featureNames,
                'Z' : Z
            }

            # print out some info about the clusters
            print 'loaded %d clusters' % clusters.shape[0]
            sum = 0
            for i in xrange( clusters.shape[0] ):
                count = numpy.sum( partition[:] == i )
                sum += count
                print 'cluster %d: %d' % ( i, count )
    
            # keep the clusters, the partition, the silhouette and
            # the intra-cluster distances as public members
            self.nonControlNormFeatures = norm_features
            self.clusters = clusters
            self.partition = partition
            self.nonControlWeights = None
            self.intraClusterDistances = None
            self.interClusterDistances = None
            self.nonControlFeatureIds = featureIds
            self.nonControlFeatureNames = featureNames
            self.nonControlClusterProfiles = clusterProfiles
            self.Z = Z
            self.method = method
            self.clusterContainer = clusterContainer

            self.clusterDist = numpy.empty( ( self.pdc.objFeatures.shape[0], clusters.shape[0] ) )
            self.clusterDist[:,:] = numpy.inf
            self.clusterDist[ self.get_non_control_cell_mask() ] = distance.minkowski_cdist( norm_features, self.clusters )

            # add the cluster ID to the object features
            tmpFeatures = self.pdc.objFeatures
            self.pdc.objFeatures = numpy.empty( ( tmpFeatures.shape[0], tmpFeatures.shape[1] + 1 ) )
            self.pdc.objFeatures[:,:-1] = tmpFeatures[:,:]
            cluster_ids = -numpy.ones( ( tmpFeatures.shape[0], ) )
            cluster_ids[ self.get_non_control_cell_mask() ] = self.partition
            self.pdc.objFeatures[:,-1] = cluster_ids[:]
            self.pdc.objFeatureIds[ 'ClusterID' ] = tmpFeatures.shape[1]
            del tmpFeatures

        finally:

            if filename != None:
                file.close()

    def get_total_cell_mask(self):
        return numpy.ones( self.validCellMask.shape, dtype=bool )
    def get_valid_cell_mask(self):
        return self.validCellMask
    def get_valid_image_mask(self):
        return self.validImageMask
    def get_control_treatment_cell_mask(self):
        return self.controlTreatmentMask
    def get_non_control_treatment_cell_mask(self):
        return numpy.invert( self.controlTreatmentMask )
    def get_non_control_cell_mask(self):
        return self.nonControlCellMask
    def get_control_cell_mask(self):
        return self.controlCellMask
    def get_invalid_cell_mask(self):
        return self.mask_not( self.validCellMask )
    def get_invalid_image_mask(self):
        return self.mask_not( self.validImageMask )
    def get_well_cell_mask(self, trId):
        return self.pdc.objFeatures[ : , self.pdc.objWellFeatureId ] == trId
    def get_well_image_mask(self, trId):
        return self.pdc.imgFeatures[ : , self.pdc.imgWellFeatureId ] == trId
    def get_treatment_cell_mask(self, trId):
        return self.pdc.objFeatures[ : , self.pdc.objTreatmentFeatureId ] == trId
    def get_treatment_image_mask(self, trId):
        return self.pdc.imgFeatures[ : , self.pdc.imgTreatmentFeatureId ] == trId
    def get_replicate_cell_mask(self, repId):
        return self.pdc.objFeatures[ : , self.pdc.objReplicateFeatureId ] == repId
    def get_replicate_image_mask(self, repId):
        return self.pdc.imgFeatures[ : , self.pdc.imgReplicateFeatureId ] == repId
    def get_slide_cell_mask(self, trId, repId):
        return self.mask_or( self.get_treatment_cell_mask( trId ), self.get_replicate_cell_mask( repId ) )
    def get_slide_image_mask(self, trId, repId):
        return self.mask_or( self.get_treatment_image_mask( trId ), self.get_replicate_image_mask( repId ) )
    def mask_not(self, mask):
        numpy.invert( mask )
    def mask_or(self, *masks):
        mask = masks[0]
        for i in xrange( 1, len( masks ) ):
            mask = numpy.logical_or( mask, masks[ i ] )
        return mask
    def mask_and(self, *masks):
        mask = masks[0]
        for i in xrange( 1, len( masks ) ):
            mask = numpy.logical_and( mask, masks[ i ] )
        return mask
    def get_cell_mask(self, *args, **kwargs):
        mask = self.get_total_cell_mask()
        for arg in args:
            arg = arg.lower()
            if arg in [ 'control', 'ctrl' ]:
                mask2 = self.get_control_cell_mask()
            elif arg in [ 'noncontrol', 'non-control', 'nonctrl', 'non-ctrl' ]:
                mask2 = self.get_non_control_cell_mask()
            elif arg == 'valid':
                mask2 = self.get_valid_cell_mask()
            elif arg == 'invalid':
                mask2 = self.get_invalid_cell_mask()
            mask = self.mask_and( mask, mask2 )
        wellId = kwargs.get( 'wellId', -1 )
        trId = kwargs.get( 'trId', -1 )
        repId = kwargs.get( 'repId', -1 )
        if wellId > -1:
            mask2 = self.get_well_cell_mask( wellId )
            mask = self.mask_and( mask, mask2 )
        if trId > -1:
            mask2 = self.get_treatment_cell_mask( trId )
            mask = self.mask_and( mask, mask2 )
        if repId > -1:
            mask2 = self.get_replicate_cell_mask( trId )
            mask = self.mask_and( mask, mask2 )
        return mask
    def get_image_mask(self, *args, **kwargs):
        mask = self.get_total_image_mask()
        for arg in args:
            arg = arg.lower()
            if arg in [ 'control', 'ctrl' ]:
                mask2 = self.get_control_image_mask()
            elif arg in [ 'noncontrol', 'non-control', 'nonctrl', 'non-ctrl' ]:
                mask2 = self.get_non_control_image_mask()
            elif arg == 'valid':
                mask2 = self.get_valid_image_mask()
            elif arg == 'invalid':
                mask2 = self.get_invalid_image_mask()
            mask = self.mask_and( mask, mask2 )
        wellId = kwargs.get( 'wellId', -1 )
        trId = kwargs.get( 'trId', -1 )
        repId = kwargs.get( 'repId', -1 )
        if wellId > -1:
            mask2 = self.get_well_cell_mask( wellId )
            mask = self.mask_and( mask, mask2 )
        if trId > -1:
            mask2 = self.get_treatment_cell_mask( trId )
            mask = self.mask_and( mask, mask2 )
        if repId > -1:
            mask2 = self.get_replicate_cell_mask( trId )
            mask = self.mask_and( mask, mask2 )
        return mask


    def save_clusters(self, file):

        #dict = { 'clusters' : self.clusters, 'featureNames' : self.nonControlFeatureNames }
        dict = { 'clusterContainer' : self.clusterContainer }

        filename = None
        if type( file ) == str:
            filename = file
            file = open( file, 'w' )

        try:

            import cPickle

            p = cPickle.Pickler( file )
            p.dump( dict )

        finally:

            if filename != None:
                file.close()


    def stop(self):
        """Stop the running pipeline thread
        """

        # this flag has to be checked regularly by the pipeline thread (e.g. through update_progress)
        self.continue_running = False


    def update_progress(self, progress):
        """Update the progress of the running pipeline thread
    
        Input parameters:
            - progress: the progress of the pipeline thread, usually between 0 and 100
        """

        # if a callback method has been specified, call it (this method must be thread-safe!)
        if self.progressCallback != None:
            self.progressCallback( progress )
        # otherwise, emit the 'updateProgress' signal (this is done asynchronously)
        else:
            self.emit( SIGNAL('updateProgress'), progress )

        # return wether the pipeline thread should stop (because the stop()-method has been called)
        return not self.has_been_stopped()


    def start_quality_control(self, progressCallback=None):
        """Start the pipeline thread to perform the quality control
    
        Input parameters:
            - progressCallback: A thread-safe callback method. This method must
              accept an integer parameter that contains the progress of the
              pipeline thread (usually between 0 and 100)
        """

        self.progressCallback = progressCallback

        # run the quality-control within the thread
        self.start_method( self.run_quality_control )

    def run_quality_control(self):
        """Perform quality control
        """

        if not self.update_progress( 0 ):
            return False

        # print out the total number of images and objects
        print 'number of images: %d' % len( self.pdc.images )
        print 'number of objects: %d' % len( self.pdc.objects )


        if not self.update_progress( 20 ):
            return False

        print 'resetting image and object states...'

        # set state of every image and object to the import-state
        for img in self.pdc.images:
            img.state = img.import_state
        for obj in self.pdc.objects:
            obj.state = obj.import_state

        print 'doing quality control...'

        # Call the method that actually performs the quality control.
        # See quality_control.quality_control() for details.
        # This gives us masks of the images and cells, that passed the quality control.
        validImageMask, validCellMask = quality_control.quality_control( self.pdc )

        # print out the number of images and cells, that passed the quality control
        print 'number of valid images: %d' % validImageMask.sum()
        print 'number of valid cells: %d' % validCellMask.sum()

        if not self.update_progress( 100 ):
            return False

        # keep the image- and cell-mask as public members
        self.validImageMask = validImageMask
        self.validCellMask = validCellMask

        # update state of the pipeline
        self.__state = self.PIPELINE_STATE_QUALITY_CONTROL

        #for tr in self.pdc.treatments:
        #    mask = self.pdc.objFeatures[ : , self.pdc.objTreatmentFeatureId ] == tr.index
        #    mask = numpy.logical_and( self.validCellMask, mask )
        #    print 'treatment %s: %d cells passed the quality control' % ( tr.name, numpy.sum( mask ) )

        # indicate success
        return True


    def start_pre_filtering(self, controlFilterMode=analyse.FILTER_MODE_MEDIAN_xMAD, featureNames=None, progressCallback=None):
        """Start the pipeline thread to perform the pre-filtering
    
        Input parameters:
            - progressCallback: A thread-safe callback method. This method must
              accept an integer parameter that contains the progress of the
              pipeline thread (usually between 0 and 100)
        """

        self.progressCallback = progressCallback

        # run the quality-control within the thread
        self.start_method( self.run_pre_filtering, controlFilterMode, featureNames )


    def run_pre_filtering(self, controlFilterMode=analyse.FILTER_MODE_MEDIAN_xMAD, featureNames=None):
        """Perform pre-filtering
        """

        if not self.update_progress( 0 ):
            return False

        print 'Running pre-filtering...'

        if featureNames == None and analyse.filter_feature_set > -1:
            print ( 'Using feature set %d:' % analyse.filter_feature_set ), self.clusterConfiguration[ analyse.filter_feature_set ][ 0 ]
            featureNames = self.clusterConfiguration[ analyse.filter_feature_set ][ 1 ]

        analyse.output_mean_control_cell_intensities( self.pdc, self.validCellMask )

        # reject cells which have NaN or Inf values
        validCellMask, featureIds, bad_feature_cell_count = analyse.reject_cells_with_bad_features( self.pdc, self.validImageMask, self.validCellMask, featureNames )

        print 'NAN:', numpy.sum(numpy.isnan(self.pdc.objFeatures[validCellMask][:, featureIds]))

        # perform a PCA on the features
        self.pca = analyse.run_pca(self.pdc, validCellMask, featureIds, PCA_components)

        num_of_cells_with_bad_features = numpy.sum( self.validCellMask ) - numpy.sum( validCellMask )

        self.validCellMask = validCellMask

        # print out the number of cells, that have no bad feature-values
        print 'number of cells with bad feature values: %d' % num_of_cells_with_bad_features

        # for each feature where cells had bad values, print out the number of those cells
        for i in xrange( len( featureIds ) ):
            featureId = featureIds[i]
            count = bad_feature_cell_count[i]
            if count > 0:
                print '  feature %d: %d bad cells' % ( featureId, count )

        # create a mask that contains all the cells in every control treatment
        #
        # we start with an empty mask
        control_treatment_mask = numpy.zeros( validCellMask.shape, dtype=bool )
        # and for each control treatment...
        for name in analyse.control_treatment_names:
            control_treatment = self.pdc.treatments[ self.pdc.treatmentByName[ name ] ]
            # we add the corresponding cells to the mask
            control_treatment_mask = self.mask_or(
                    control_treatment_mask,
                    self.pdc.objFeatures[ : , self.pdc.objTreatmentFeatureId ] == control_treatment.index
            )
        self.controlTreatmentMask = control_treatment_mask

        # reject control cells and select the features to be used
        controlCellMask, nonControlCellMask, featureIds, mahal_dist, mahal_cutoff_window = \
            analyse.reject_control_cells(
                self.pdc, self.validImageMask, self.validCellMask, self.controlTreatmentMask,
                featureIds, controlFilterMode, self.pca)
        #controlCellMask, nonControlCellMask, featureIds = analyse.cutoff_control_cells( self.pdc, featureNames, validImageMask, validCellMask )

        # TODO
        self.mahal_cutoff_window = mahal_cutoff_window

        # reject control-like treatments
        nonControlTreatments = analyse.reject_control_treatments(self.pdc, self.validCellMask, nonControlCellMask)

        #print 'features used for phenotypic filtering:'
        #fNames = self.pdc.objFeatureIds.keys()
        #fValues = self.pdc.objFeatureIds.values()
        #for featureId in featureIds:
            #i = fValues.index( featureId )
            #featureName = fNames[ i ]
            #print '  %d -> %s' % ( featureId, featureName )
        #print

        # print out some info
        print 'number of features used for phenotypic filtering: %d' % len(featureIds)

        if controlCellMask == None:
            return False

        def append_object_feature(name, data):
            tmpFeatures = self.pdc.objFeatures
            self.pdc.objFeatures = numpy.empty( ( tmpFeatures.shape[0], tmpFeatures.shape[1] + 1 ) )
            self.pdc.objFeatures[:,:-1] = tmpFeatures[:,:]
            self.pdc.objFeatures[:,-1] = data
            self.pdc.objFeatureIds[ name ] = tmpFeatures.shape[1]
            del tmpFeatures
        # add the mahalanobis distance to the object features
        append_object_feature('Mahalanobis Distance', mahal_dist)

        #TODO
        features = self.pdc.objFeatures[validCellMask][:, featureIds]
        pca_features = self.pca.transform(features)
        new_pca_features = numpy.empty((self.pdc.objFeatures.shape[0], pca_features.shape[1]))
        new_pca_features[validCellMask] = pca_features
        new_pca_features[numpy.invert(validCellMask)] = numpy.inf
        pca_features = new_pca_features
        for i in xrange(pca_features.shape[1]):
            append_object_feature('PCA %d' % (i+1), pca_features[:,i])

        ## add the sqrt of mahalanobis distance to the object features
        #tmpFeatures = self.pdc.objFeatures
        #self.pdc.objFeatures = numpy.empty( ( tmpFeatures.shape[0], tmpFeatures.shape[1] + 1 ) )
        #self.pdc.objFeatures[:,:-1] = tmpFeatures[:,:]
        #self.pdc.objFeatures[:,-1] = numpy.sqrt( mahal_dist[:] )
        #self.pdc.objFeatureIds[ 'Mahalanobis Distance Sqrt' ] = tmpFeatures.shape[1]
        #del tmpFeatures

        # reject manually selected treatments
        for tr_name in reject_treatments:
            tr_id = self.pdc.treatmentByName[ tr_name ]
            tr = self.pdc.treatments[ tr_id ]
            tr_mask = self.pdc.objFeatures[ :, self.pdc.objTreatmentFeatureId ] == tr.index
            tmp_mask = numpy.logical_and( nonControlCellMask, tr_mask )
            controlCellMask[ tmp_mask ] = True
            nonControlCellMask[ tr_mask ] = False

        # create a list of treatments which are not in 'reject-treatments'
        validTreatments = []
        validTreatmentIds = []
        for tr in self.pdc.treatments:
            if tr.name not in reject_treatments:
                validTreatments.append( tr )
                validTreatmentIds.append( tr.index )

        # print out the number of cells to be used for clustering
        print 'cells used for clustering: %d' % numpy.sum( nonControlCellMask )

        # extract the features of the non-control-like cells
        nonControlFeatures = self.pdc.objFeatures[ nonControlCellMask ]

        nonControlMahalDist = mahal_dist[ nonControlCellMask ]

        if not self.update_progress( 100 ):
            return False

        # keep the cell masks and the features of the non-control-like cells as public members
        self.controlCellMask = controlCellMask
        self.nonControlCellMask = nonControlCellMask
        self.nonControlFeatures = nonControlFeatures
        self.nonControlMahalDist = nonControlMahalDist
        self.featureIds = featureIds

        # keep the list of valid treatments
        self.validTreatments = validTreatments
        self.validTreatmentIds = validTreatmentIds

        # update state of the pipeline
        self.__state = self.PIPELINE_STATE_PRE_FILTERING

        # indicate success
        return True


    #def run_pipeline(self, featureset_index=0, progressCallback=None):
    #    """Start the pipeline thread to perform the pre-filtering
    #
    #    Input parameters:
    #        - progressCallback: A thread-safe callback method. This method must
    #          accept an integer parameter that contains the progress of the
    #          pipeline thread (usually between 0 and 100)
    #    """
    #
    #    self.progressCallback = progressCallback
    #
    #    if not self.update_progress( 0 ):
    #        return False
    #
    #    print 'number of images: %d' % len( self.pdc.images )
    #    print 'number of objects: %d' % len( self.pdc.objects )
    #
    #
    #    if not self.update_progress( 20 ):
    #        return False
    #
    #
    #    print 'doing quality control'
    #
    #    # do some quality control of the images and cells
    #    validImageMask, validCellMask = quality_control.quality_control( self.pdc )
    #
    #
    #    if not self.update_progress( 70 ):
    #        return False
    #
    #
    #    #imgFeatures = pdc.imgFeatures[ validImageMask ]
    #    #objFeatures = pdc.objFeatures[ validCellMask ]
    #
    #    print 'number of valid images: %d' % validImageMask.sum()
    #    print 'number of valid cells: %d' % validCellMask.sum()
    #
    #
    #    featureNames = self.clusterConfiguration[ featureset_index ][ 1 ]
    #
    #    print featureNames
    #
    #    # cutoff control cells and select the features to be used
    #    controlCellMask, nonControlCellMask, featureIds = analyse.cutoff_control_cells( self.pdc, featureNames, validImageMask, validCellMask )
    #
    #    if controlCellMask == None:
    #        return False
    #
    #    if not self.update_progress( 80 ):
    #        return False
    #
    #
    #    featureNames = list( featureIds )
    #    for k,v in self.pdc.objFeatureIds.iteritems():
    #        if v in featureNames:
    #            i = featureNames.index( v )
    #            if i >= 0:
    #                featureNames[ i ] = k
    #
    #
    #    nonControlFeatures = self.pdc.objFeatures[ nonControlCellMask ]
    #
    #    # transform features into mahalanobis space and calculate the transformation matrix
    #    mahalTransformation = distance.mahalanobis_transformation( nonControlFeatures[ : , featureIds ] )
    #
    #    # transformate data points into mahalanobis space
    #    nonControlTransformedFeatures = distance.transform_features( nonControlFeatures[ : , featureIds ], mahalTransformation )
    #
    #    #print 'Using the following features:'
    #    #print '\n'.join(mahalFeatureNames)
    #    print 'using %d features' % len(featureIds)
    #
    #    if not self.update_progress( 100 ):
    #        return False
    #
    #
    #    """self.update_progress( 50 )
    #
    #
    #    global progress
    #    progress = 50.0
    #    def cluster_callback(iterations, swaps):
    #        global progress
    #        progress += 0.5
    #        self.update_progress( int( progress + 0.5 ) )
    #        return True
    #
    #    # cluster objects in mahalanobis space
    #    try:
    #        number_of_clusters
    #    except:
    #        number_of_clusters = len( self.pdc.treatments )
    #
    #    partition,clusters = cluster.cluster_by_dist( nonControlTransformedFeatures, number_of_clusters, clustering_swap_threshold, cluster_callback )
    #
    #    #partition,clusters = cluster.cluster_by_fermi_dirac_dist(
    #    #                                nonControlTransformedFeatures,
    #    #                                number_of_clusters,
    #    #                                clustering_swap_threshold,
    #    #                                fermi_dirac_sim_sharpness,
    #    #                                cluster_callback
    #    #)
    #
    #
    #    self.update_progress( 100 )
    #
    #
    #    self.clusters = clusters
    #    self.partition = partition"""
    #
    #    self.validImageMask = validImageMask
    #    self.validCellMask = validCellMask
    #
    #    self.controlCellMask = controlCellMask
    #    self.nonControlCellMask = nonControlCellMask
    #
    #    self.featureIds = featureIds
    #    self.featureNames = featureNames
    #
    #    self.nonControlFeatures = nonControlFeatures
    #    self.nonControlTransformedFeatures = nonControlTransformedFeatures
    #
    #    return True


    def prepare_clustering(self, max_num_of_clusters=-1, progressCallback=None):

        if ( max_num_of_clusters == None ) or max_num_of_clusters <= 0:
            
            # cluster objects in mahalanobis space
            try:
                number_of_clusters
            except:
                number_of_clusters = len( self.pdc.treatments )

            max_num_of_clusters = number_of_clusters


        self.progressCallback = progressCallback

        if not self.update_progress( 0 ):
            return False

        global progress
        progress = 0.0
        def cluster_callback(iterations, swaps):
            global progress
            progress += 2
            return self.update_progress( int( progress + 0.5 ) )

        best_num_of_clusters, gaps, sk = cluster.determine_num_of_clusters( self.nonControlTransformedFeatures, max_num_of_clusters, num_of_reference_datasets )

        if not self.update_progress( 100 ):
            return False

        return best_num_of_clusters, gaps, sk


    def start_clustering(self, group_description, method, index=0, param1=-1, param2=-1, param3=-1, param4=-1, exp_factor=-1, calculate_silhouette=False, progressCallback=None):
        """Start the pipeline thread to perform the clustering
    
        Input parameters:
            - index: The index of the featureset to use
            - num_of_clusters: The number of clusters to be used.
              if the value -1 is passed, an educated guess is made
            - calculate_silhouette: Determins wether to calculate the silhouette
              of the final clustering
            - progressCallback: A thread-safe callback method. This method must
              accept an integer parameter that contains the progress of the
              pipeline thread (usually between 0 and 100)
        """

        self.progressCallback = progressCallback

        # run the quality-control within the thread
        self.start_method( self.run_clustering, group_description, method, index, param1, param2, param3, param4, exp_factor, calculate_silhouette )

    def write_features_file(self, filename, features, id_mapping=None, featureNames=None):
        f = open(filename,'w')

        if id_mapping == None:
            id_mapping = numpy.arange( features.shape[1] )

        row = []
        row.append( '' )
        for i in xrange( features.shape[1] ):
            row.append('%d' % id_mapping[i])
        f.write( ','.join( row ) + '\n' )
        row = []
        row.append( '' )
        for i in xrange( features.shape[1] ):
            row.append( featureNames[i] )
        f.write( ','.join( row ) + '\n' )
        feat = features
        for i in xrange( feat.shape[0] ):
            row = []
            row.append( '%d' % i )
            for j in xrange( feat.shape[1] ):
                row.append( '%f' % ( feat[ i, j ] ) )
            f.write( ','.join( row ) + '\n' )
        f.close()


    def __compute_normalized_features(self, cellMask, featureNames, controlCellMask=None, num_of_features=10):

        if controlCellMask == None:
            controlCellMask = cellMask

        featureIds = []
        for featureName in featureNames:
            featureIds.append( self.pdc.objFeatureIds[ featureName ] )

        featureIds = numpy.array( featureIds )
        print 'featureIds:', featureIds

        # extract the features for the clustering
        features = self.pdc.objFeatures[ cellMask ][ : , featureIds ]
        controlFeatures = self.pdc.objFeatures[ controlCellMask ][ : , featureIds ]
        print 'features.shape:', features.shape

        nan_mask = numpy.isnan( controlFeatures )
        collapsed_nan_mask = numpy.any( nan_mask, axis=1 )
        inverse_mask = numpy.invert( collapsed_nan_mask )
        controlFeatures = controlFeatures[ inverse_mask ]

        # normalize the features
        #
        # calculate medians
        #median = numpy.median( features, 0 )
        # calculate median-absolute-deviations
        #mad = numpy.median( numpy.abs( features - median ), 0 )
        # calculate the normalized features
        #norm_features = ( features - median ) / mad
        # calculate standard-deviations
        stddev = numpy.std( controlFeatures, 0 )
        # calculate means
        mean = numpy.mean( controlFeatures, 0 )
        # calculate the normalized features
        norm_features = ( features - mean ) / stddev
        # create a mask of valid features
        nan_mask = numpy.isnan( norm_features )
        inf_mask = numpy.isinf( norm_features )
        invalid_feature_nan_mask = numpy.all(nan_mask, 0)
        invalid_feature_inf_mask = numpy.all(inf_mask, 0)
        invalid_feature_mask = numpy.logical_or( invalid_feature_nan_mask, invalid_feature_inf_mask )
        valid_feature_mask = numpy.logical_not( invalid_feature_mask )
        print 'sum(invalid_feature_nan_mask):', numpy.sum(invalid_feature_nan_mask)
        print 'sum(invalid_feature_inf_mask):', numpy.sum(invalid_feature_inf_mask)

        norm_features = norm_features[ : , valid_feature_mask ]

        nan_mask = numpy.isnan( norm_features )
        inf_mask = numpy.isinf( norm_features )
        invalid_cell_nan_mask = numpy.any(nan_mask,1)
        invalid_cell_inf_mask = numpy.any(inf_mask,1)
        invalid_cell_mask = numpy.logical_or(invalid_cell_nan_mask, invalid_cell_inf_mask)
        valid_cell_mask = numpy.logical_not(invalid_cell_mask)
        print 'invalid nan cells:', numpy.sum(invalid_cell_nan_mask)
        print 'invalid inf cells:', numpy.sum(invalid_cell_inf_mask)

        # only keep valid features
        norm_features = norm_features[valid_cell_mask]

        return valid_cell_mask, norm_features, valid_feature_mask, featureIds[ valid_feature_mask ]

    def compute_normalized_features(self, cellMask, featureset_index, controlCellMask=None):

        # determine the IDs of the features to be used
        featureNames = self.clusterConfiguration[ featureset_index ][ 1 ]
        print 'featureNames:', featureNames
        return self.__compute_normalized_features( cellMask, featureNames, controlCellMask )


    """def run_histogram_analysation(self, featureset_index=0):

        histograms = numpy.empty( (  ) )

        for tr in self.pdc.treatments:
            for repl in tr.replicates:"""


    def run_clustering(self, group_description, method, index=0, param1=-1, param2=-1, param3=-1, param4=-1, exp_factor=-1, calculate_silhouette=False):
        """Perform the clustering
    
        Input parameters:
            - method: the clustering method to use (see cluster.get_hcluster_methods)
            - featureset_index: The index of the feature-set to use
            - param1: first parameter for the clustering (see cluster.hcluster)
            - param2: second parameter for the clustering (see cluster.hcluster)
            - calculate_silhouette: Determins wether to calculate the silhouette
              of the final clustering
        """

        print('param1:', param1, 'param2:', param2, 'param3:', param3,
              'param4:', param4, 'exp_factor:', exp_factor)

        featureset_index = index

        # If num_of_clusters is not provided, check wether the parameter
        # number_of_clusters has been defined.
        # Otherwise make an 'educated' guess
        """if ( num_of_clusters == None ) or num_of_clusters <= 0:

            try:
                number_of_clusters
            except:
                number_of_clusters = len( self.pdc.treatments )

            num_of_clusters = number_of_clusters"""

        if not self.update_progress( 0 ):
            return False

        # determine the IDs of the features to be used
        featureNames = self.clusterConfiguration[ featureset_index ][ 1 ]
        featureIds = []
        for featureName in featureNames:
            featureIds.append( self.pdc.objFeatureIds[ featureName ] )

        id_mapping = numpy.array( featureIds )

        # extract the features for the clustering
        features = self.pdc.objFeatures[ self.nonControlCellMask ][ : , featureIds ]

        # keep the extracted features used for clustering as a public member
        self.nonControlClusterFeatures = features


        # defines a progress callback for the clustering method
        global progress
        progress = 0.0
        def cluster_callback(iterations, swaps):
            global progress
            progress += 2
            return self.update_progress( int( progress + 0.5 ) )

        """# normalize the features
        #
        # calculate medians
        #median = numpy.median( features, 0 )
        # calculate median-absolute-deviations
        #mad = numpy.median( numpy.abs( features - median ), 0 )
        # calculate the normalized features
        #norm_features = ( features - median ) / mad
        # calculate standard-deviations
        stddev = numpy.std( features, 0 )
        # calculate means
        mean = numpy.mean( features, 0 )
        # calculate the normalized features
        norm_features = ( features - mean ) / stddev
        # create a mask of valid features
        nan_mask = numpy.isnan( norm_features )
        inf_mask = numpy.isinf( norm_features )
        invalid_mask = numpy.logical_or( nan_mask, inf_mask )
        invalid_mask = numpy.any( invalid_mask, 0 )
        valid_mask = numpy.logical_not( invalid_mask )"""


        """f = open('/home/hepp/featureIds.xls','w')
        fids = self.pdc.objFeatureIds
        keys = fids.keys()
        keys.sort()
        for key in keys:
            f.write('%s,%d\n' % ( key, fids[ key ] ) )
        f.close()

        #featureNames = self.pdc.objFeatureIds.keys()
        #featureNames.sort()
        self.write_features_file( '/home/hepp/features.xls', features, id_mapping, featureNames )
        self.write_features_file( '/home/hepp/norm_features.xls', norm_features, id_mapping, featureNames )"""


        # only keep valid features
        #norm_features = norm_features[ : , valid_feature_mask ]

        print 'group_description:', group_description
        groups = grouping.get_groups(group_description, self.pdc)
        #names,masks = zip(*groups)
        groupingCellMask = grouping.join_groups(groups)
        groupingCellMask = self.mask_and(groupingCellMask, self.nonControlCellMask)

        valid_cell_mask, norm_features, valid_feature_mask, newFeatureIds = self.compute_normalized_features( groupingCellMask, featureset_index, self.get_control_treatment_cell_mask() )
        featureIds = newFeatureIds
        features = features[valid_cell_mask][:,valid_feature_mask]
        cellMask = groupingCellMask.copy()
        tempMask = cellMask[cellMask]
        tempMask[numpy.logical_not(valid_cell_mask)] = False
        cellMask[cellMask] = tempMask
        #print 'control_treatment_cell_maskk:', self.get_control_treatment_cell_mask().shape, self.get_control_treatment_cell_mask().dtype
        #print 'nonControlCellMask:', self.nonControlCellMask.shape, self.nonControlCellMask.dtype

        #a = numpy.arange(cellMask.shape[0])
        #a = a[cellMask]
        #numpy.random.shuffle(a)
        #a = a[:5000]
        #m = cellMask.copy()
        #m[:] = False
        #m[a] = True
        #m = m[cellMask]
        #norm_features = norm_features[m]
        #valid_cell_mask = valid_cell_mask[m]
        #cellMask[:] = False
        #cellMask[a] = True

        print 'norm_features.shape:', norm_features.shape
        print 'numpy.sum(valid_feature_mask):', numpy.sum(valid_feature_mask)
        print 'valid_feature_mask.shape:', valid_feature_mask.shape
        print 'newFeatureIds:', newFeatureIds

        best_feature_mask = numpy.ones( ( norm_features.shape[1], ), dtype=bool )


        do_feature_selection = False
        if do_feature_selection:

            from batch import utils as batch_utils

            NUM_OF_FEATURES = 5
            BOOTSTRAP_COUNT_MULTIPLIER = 0.1
            BOOTSTRAP_COUNT_MAX = 200
            BOOTSTRAP_SAMPLE_SIZE_RATIO = 0.2
            #RESAMPLE_MAX = 3

            ctrl_mask = self.mask_and( self.get_control_treatment_cell_mask(), self.get_valid_cell_mask() )
            obs = self.pdc.objFeatures[ ctrl_mask ][ : , newFeatureIds ]
            mask3 = self.mask_and( self.get_non_control_treatment_cell_mask(), self.get_valid_cell_mask() )
            bootstrap_sample_size = round( BOOTSTRAP_SAMPLE_SIZE_RATIO * obs.shape[0] )
            obs1 = numpy.empty( ( bootstrap_sample_size, newFeatureIds.shape[0] ) )
            #self.pdc.objFeatures[ mask1 ][ : , newFeatureIds ]
            #edf1 = batch_utils.compute_edf( obs1 )
            obs3 = self.pdc.objFeatures[ mask3 ][ : , newFeatureIds ]
            edf3 = batch_utils.compute_edf( obs3 )
            obs2 = numpy.empty( obs1.shape )
            bootstrap_count = int( BOOTSTRAP_COUNT_MULTIPLIER * obs.shape[0] + 0.5 )
            bootstrap_count = numpy.min( [ bootstrap_count, BOOTSTRAP_COUNT_MAX ] )
            dist = numpy.empty( ( bootstrap_count + 1, norm_features.shape[1] ) )
            bootstrap_dist = numpy.empty( ( bootstrap_count, norm_features.shape[1] ) )
            print 'bootstrapping ks statistics (%d)...' % ( bootstrap_count )
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
            edf = batch_utils.compute_edf( obs )
            for k in xrange( norm_features.shape[1] ):
                support = edf[ k ]
                support3 = edf3[ k ]
                dist[ bootstrap_count, k ] = batch_utils.compute_edf_distance( support, support3 )

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
                if num_of_features > 0 and feature_quality.shape[0] > num_of_features:
                    num_of_features = min( num_of_features, feature_quality.shape[0] )
                    sorted_feature_indices = numpy.argsort( feature_quality )
                    best_feature_mask[ sorted_feature_indices[ : -num_of_features ] ] = False
                return best_feature_mask

            print 'selecting %d best features...' % NUM_OF_FEATURES

            feature_quality = dist[-1] - mean_bootstrap_dist
            best_feature_mask = select_best_features( feature_quality, NUM_OF_FEATURES )
            norm_features = norm_features[ : , best_feature_mask ]
            newFeatureIds = newFeatureIds[ best_feature_mask ]
            #ids = numpy.arange( valid_feature_mask.shape[0] )[ valid_feature_mask ][ best_feature_mask ]
            #valid_feature_mask = numpy.zeros( valid_feature_mask.shape, dtype=bool )
            #valid_feature_mask[ ids ] = True
            #ids = fids[ valid_feature_mask ]
            valid_feature_mask2 = numpy.zeros( valid_feature_mask.shape, dtype=bool )
            valid_feature_mask2[ valid_feature_mask ] = best_feature_mask
            valid_feature_mask = valid_feature_mask2

            print 'best informative features:'
            print newFeatureIds
            for (i,feature_id) in enumerate( newFeatureIds ):
                fname = None
                for fn,fid in self.pdc.objFeatureIds.iteritems():
                    if fid == feature_id:
                        fname = fn
                        break
                print 'feature (%d): %s, %f' % ( feature_id, fname, feature_quality[i] )


        #print 'features used for clustering:'
        #fNames = self.pdc.objFeatureIds.keys()
        #fValues = self.pdc.objFeatureIds.values()
        #for featureId in newFeatureIds:
            #i = fValues.index( featureId )
            #featureName = fNames[ i ]
            #print '  %d -> %s' % ( featureId, featureName )
        #print

        # print out some info
        print 'number of features used for clustering: %d' % norm_features.shape[ 1 ]
        print 'number of features not used for clustering: %d' % ( features.shape[ 1 ] - norm_features.shape[ 1 ] )

        # Call the actual clustering-routine.
        # Use K-Means and cluster the non-control-features
        # (see cluster.cluster for details)
        #partition,clusters = cluster.cluster_by_dist( self.nonControlTransformedFeatures, num_of_clusters, clustering_swap_threshold, cluster_callback )
        #partition,clusters,silhouette,weights = cluster.cluster(
        #        cluster.CLUSTER_METHOD_KMEANS,
        #        norm_features,
        #        num_of_clusters,
        #        minkowski_p,
        #        calculate_silhouette,
        #        cluster_callback
        #)
        """partition,clusters = cluster.cluster_by_fermi_dirac_dist(
                                        nonControlTransformedFeatures,
                                        num_of_clusters,
                                        clustering_swap_threshold,
                                        fermi_dirac_sim_sharpness,
                                        cluster_callback
        )"""

        if param3 == -1:
            param3 = minkowski_p

        #partition,clusters,Z = cluster.cluster_hierarchical_seeds( norm_features, num_of_clusters, objects_to_cluster, minkowski_p )
        #dist_threshold = 15.0
        #partition,clusters,Z = cluster.hcluster( method, norm_features, param1, param2, featureset_index, param3 )
        if method == 'mahalanobis':
            method2 = 'mahalanobis'
            method = 'k-means'

        t1 = time.time()
        c1 = time.clock()
        print 't1: %.2f' % t1
        print 'c1: %.2f' % c1

        import cPickle
        f = open('/g/pepperkok/hepp/cell_objects.pic', 'w')
        p = cPickle.Pickler(f)
        p.dump(norm_features)
        f.close()

        partition,clusters,Z = cluster.hcluster_special( method, norm_features, param1, param2, param4, featureset_index, param3 )

        c2 = time.clock()
        t2 = time.time()
        print 't2: %.2f' % t2
        print 'c2: %.2f' % c2

        dc = c2 - c1
        dt = t2 - t1
        print 'clocks: %.2f' % dc
        print 'time: %.2f' % dt

        #if clusters == None:
        #    clusters = cluster.compute_centroids_from_partition(norm_features, partition)

        self.Z = Z
        try:
            method = method2
        except:
            pass

        print partition.shape
        if method not in [ 'kd-tree', 'random' ]:
            print clusters.shape
            print Z.shape

        import cPickle
        clusterSizes = numpy.empty((clusters.shape[0],), dtype=int)
        for i in xrange(clusters.shape[0]):
            clusterSizes[i] = numpy.sum(partition == i)
        # sanity check
        if numpy.sum(clusterSizes) != norm_features.shape[0]:
            raise Exception('Something went really wrong here!')
        f = open('/g/pepperkok/hepp/cluster_sizes.pic', 'w')
        p = cPickle.Pickler(f)
        p.dump(clusterSizes)
        f.close()

        """import pickle
        f = open('/home/hepp/Z.pickle','w')
        pickle.dump( Z, f )
        f.close()
        f = open('/home/hepp/Z.xls','w')
        f.write( 'index1,index2,dist,count\n' )
        for i in xrange( Z.shape[0] ):
            f.write( '%f,%f,%f,%f\n' % ( Z[i,0],Z[i,1],Z[i,2],Z[i,3] ) )
        f.close()"""

        #weights = cluster.compute_feature_weighting( partition, norm_features, clusters, minkowski_p )
        weights = None


        if weights != None and len( weights.shape ) == 1:
            oldWeights = weights
            weights = numpy.empty( ( clusters.shape[0], weights.shape[0] ) )
            for k in xrange( clusters.shape[0] ):
                weights[k] = oldWeights


        if weights != None:

            k = 0
            p_mask = partition[:] == k

            while numpy.sum( p_mask ) == 0:
                k += 1
                p_mask = partition[:] == k

            stddevs = numpy.std( norm_features[ p_mask ], axis=0)
            all_stddevs = numpy.std( norm_features, axis=0 )

            medians = numpy.median( norm_features[ p_mask ], axis=0 )
            mads = numpy.median( numpy.abs( norm_features[ p_mask ] - medians ), axis=0 )

            print 'weights of cluster %d:' % k
            identity_arr = numpy.arange( features.shape[1] )
            mapping = identity_arr[ valid_feature_mask ]
            for i in xrange( weights.shape[0] ):
                print '%d -> %f (%f, %f) {%f, %f} [%s]' % ( id_mapping[ mapping[i] ], weights[0, i], stddevs[i], all_stddevs[i], medians[i], mads[i], featureNames[ mapping[i] ] )
            print

        #j = 120
        #for k in xrange( clusters.shape[0] ):
        #    print 'cluster %d:' % k
        #    p_mask = partition[:] == k
        #    for i in xrange( norm_features[ p_mask ].shape[0] ):
        #        print '  %f' % ( norm_features[ p_mask ][i,j] )
        #print

        """if False:

            f = open('/home/hepp/clusters.xls','w')

            row = []
            row.append( '' )
            row.append( '' )
            identity_arr = numpy.arange( features.shape[1] )
            mapping = identity_arr[ valid_feature_mask ]
            for i in xrange( norm_features.shape[1] ):
                row.append( '%d' % id_mapping[ mapping[i] ] )
            f.write( ','.join( row ) + '\n' )
            row = []
            row.append( '' )
            row.append( '' )
            for i in xrange( norm_features.shape[1] ):
                row.append( '%d' % id_mapping[i] )
            f.write( ','.join( row ) + '\n' )
            row = []
            row.append( '' )
            row.append( '#' )
            for i in xrange( norm_features.shape[1] ):
                row.append( featureNames[ mapping[i] ] )
            f.write( ','.join( row ) + '\n' )
            for k in xrange( clusters.shape[0] ):
                p_mask = partition[:] == k
                feat = features[:,valid_feature_mask][p_mask]
                sfeat = numpy.std( feat, axis=0 )
                row = []
                row.append( '%d' % k )
                row.append( '%d' % numpy.sum( p_mask ) )
                for j in xrange( sfeat.shape[0] ):
                    row.append( '%f' % ( sfeat[ j ] ) )
                f.write( ','.join( row ) + '\n' )

            f.write('\n')

            row = []
            row.append( '' )
            row.append( '' )
            identity_arr = numpy.arange( features.shape[1] )
            mapping = identity_arr[ valid_feature_mask ]
            for i in xrange( norm_features.shape[1] ):
                row.append( '%d' % id_mapping[ mapping[i] ] )
            f.write( ','.join( row ) + '\n' )
            row = []
            row.append( '' )
            row.append( '' )
            for i in xrange( norm_features.shape[1] ):
                row.append('%d' % id_mapping[i])
            f.write( ','.join( row ) + '\n' )
            row = []
            row.append( '' )
            row.append( 'k' )
            for i in xrange( norm_features.shape[1] ):
                row.append( featureNames[ mapping[i] ] )
            f.write( ','.join( row ) + '\n' )
            for k in xrange( clusters.shape[0] ):
                p_mask = partition[:] == k
                feat = features[:,valid_feature_mask][p_mask]
                for i in xrange( feat.shape[0] ):
                    row = []
                    row.append( '%d' % i )
                    row.append( '%d' % k )
                    for j in xrange( feat.shape[1] ):
                        row.append( '%f' % ( feat[ i, j ] ) )
                    f.write( ','.join( row ) + '\n' )
            f.close()

            f = open('/home/hepp/norm_clusters.xls','w')

            row = []
            row.append( '' )
            row.append( '' )
            identity_arr = numpy.arange( features.shape[1] )
            mapping = identity_arr[ valid_feature_mask ]
            for i in xrange( norm_features.shape[1] ):
                row.append( '%d' % id_mapping[ mapping[i] ] )
            f.write( ','.join( row ) + '\n' )
            row = []
            row.append( '' )
            row.append( '' )
            for i in xrange( norm_features.shape[1] ):
                row.append( '%d' % id_mapping[i] )
            f.write( ','.join( row ) + '\n' )
            row = []
            row.append( '' )
            row.append( '#' )
            for i in xrange( norm_features.shape[1] ):
                row.append( featureNames[ mapping[i] ] )
            f.write( ','.join( row ) + '\n' )
            for k in xrange( clusters.shape[0] ):
                p_mask = partition[:] == k
                feat = features[:,valid_feature_mask][p_mask]
                nfeat = norm_features[p_mask]
                sfeat = numpy.std( feat, axis=0 )
                snfeat = numpy.std( nfeat, axis=0 )
                row = []
                row.append( '%d' % k )
                row.append( '%d' % numpy.sum( p_mask ) )
                for j in xrange( sfeat.shape[0] ):
                    row.append( '%f' % ( snfeat[ j ] ) )
                f.write( ','.join( row ) + '\n' )

            f.write('\n')

            row = []
            row.append( '' )
            row.append( '' )
            identity_arr = numpy.arange( features.shape[1] )
            mapping = identity_arr[ valid_feature_mask ]
            for i in xrange( norm_features.shape[1] ):
                row.append( '%d' % id_mapping[ mapping[i] ] )
            f.write( ','.join( row ) + '\n' )
            row = []
            row.append( '' )
            row.append( '' )
            for i in xrange( norm_features.shape[1] ):
                row.append('%d' % id_mapping[i])
            f.write( ','.join( row ) + '\n' )
            row = []
            row.append( '' )
            row.append( 'k' )
            for i in xrange( norm_features.shape[1] ):
                row.append( featureNames[ mapping[i] ] )
            f.write( ','.join( row ) + '\n' )
            for k in xrange( clusters.shape[0] ):
                p_mask = partition[:] == k
                feat = features[:,valid_feature_mask][p_mask]
                nfeat = norm_features[p_mask]
                for i in xrange( nfeat.shape[0] ):
                    row = []
                    row.append( '%d' % i )
                    row.append( '%d' % k )
                    for j in xrange( nfeat.shape[1] ):
                        row.append( '%f' % ( nfeat[ i, j ] ) )
                    f.write( ','.join( row ) + '\n' )
            f.close()"""


        """partition_mask = partition >= 0
        non_partition_mask = partition < 0
        new_partition = cluster.partition_along_clusters(
            norm_features[ non_partition_mask ],
            norm_features[ partition_mask ],
            partition[ partition_mask ],
            clusters,
            None,
            param3
        )
        partition[ non_partition_mask ] = new_partition"""

        dump_cluster_index = -1

        if method in [ 'ward', 'single', 'average', 'complete' ]:
            method = 'k-means'

        if method in [ 'ward', 'single', 'average', 'complete' ]:

            discard_cluster_threshold = param4
    
            print 'discard_cluster_threshold=%d' % discard_cluster_threshold
    
    
            # Discard clusters which are smaller than discard_cluster_threshold.
            # All discarded cells will be put into a 'dump' cluster.
            keep_cluster_indices = []
            discard_cluster_indices = []
            cluster_mask = numpy.ones( ( clusters.shape[0], ), dtype=bool )
            for i in xrange( clusters.shape[ 0 ] ):
                # determine the mask of this cluster...
                partition_mask = partition[:] == i
                # and it's size...
                cluster_size = numpy.sum( partition_mask )
                # and check if it should be discarded
                if cluster_size < discard_cluster_threshold:
                    print 'discarding cluster %d: %d cells' % ( i, cluster_size )
                    partition[ partition_mask ] = -1
                    discard_cluster_indices.append( i )
                    cluster_mask[ i ] = False
    
            print 'discarded %d clusters' % ( len( discard_cluster_indices ) )
    
            clusters = clusters[ cluster_mask ]
    
            discard_cluster_indices.reverse()
            for i in discard_cluster_indices:
                mask = partition > i
                partition[ mask ] -= 1
    
            mask = partition == -1
            #partition[ mask ] = clusters.shape[0]
            #dump_cluster_index = clusters.shape[0]

            #new_clusters = numpy.empty( ( clusters.shape[0] + 1, clusters.shape[1] ) )
            #new_clusters[ : -1 ] = clusters
            #new_clusters[ -1 ] = numpy.nan
            #clusters = new_clusters

            print 'discarded %d cells' % ( numpy.sum( mask ) )
    
            print clusters.shape
    
            clusters = cluster.compute_centroids_from_partition( norm_features, partition )
    
            mask = partition >= 0
            clusterContainer = {
                'points' : norm_features[ mask ].copy(),
                'partition' : partition[ mask ].copy(),
                'clusters' : clusters.copy(),
                'dump_cluster_index' : dump_cluster_index,
                'mask' : mask.copy(),
                'featureNames' : featureNames,
                'Z' : Z,
                'method' : method,
                'featureset_index' : featureset_index,
                'exp_factor' : exp_factor,
                'param1' : param1,
                'param2': param2,
                'param3': param3,
                'param4': param4,
                'valid_feature_mask' : valid_feature_mask,
                'newFeatureIds' : newFeatureIds,
                'best_feature_mask' : best_feature_mask
            }

            # Compute standard-deviation vector for each cluster
            stddevs = numpy.empty( clusters.shape )
            for i in xrange( clusters.shape[ 0 ] ):
                partition_mask = partition[:] == i
                stddevs[ i ] = numpy.std( norm_features[ partition_mask ], axis=0 )
    
            # print out some info about the dissolving
            print 'found %d clusters' % clusters.shape[0]
            sum = 0
            for i in xrange( clusters.shape[0] ):
                count = numpy.sum( partition[:] == i )
                sum += count
                print 'cluster %d: %d' % ( i, count )
    
            print 'clustered %d out of %d cells' % ( sum, norm_features.shape[0] )
    
            print 'clusters.shape:', clusters.shape
            print 'min(partition):', numpy.min( partition ), 'max(partition):', numpy.max( partition )
    
        elif method not in [ 'kd-tree', 'random' ]:

            #if method == 'ward':
            discard_cluster_threshold = param4
    
            print 'discard_cluster_threshold=%d' % discard_cluster_threshold
    
    
            # Discard clusters which are smaller than discard_cluster_threshold.
            # All discarded cells will be put into a 'dump' cluster.
            keep_cluster_indices = []
            discard_cluster_indices = []
            cluster_mask = numpy.ones( ( clusters.shape[0], ), dtype=bool )
            for i in xrange( clusters.shape[ 0 ] ):
                # determine the mask of this cluster...
                partition_mask = partition[:] == i
                # and it's size...
                cluster_size = numpy.sum( partition_mask )
                # and check if it should be discarded
                if cluster_size < discard_cluster_threshold:
                    print 'discarding cluster %d: %d cells' % ( i, cluster_size )
                    partition[ partition_mask ] = -1
                    discard_cluster_indices.append( i )
                    cluster_mask[ i ] = False
    
            print 'discarded %d clusters' % ( len( discard_cluster_indices ) )
    
            clusters = clusters[ cluster_mask ]
    
            discard_cluster_indices.reverse()
            for i in discard_cluster_indices:
                mask = partition > i
                partition[ mask ] -= 1
    
            mask = partition == -1
            #partition[ mask ] = clusters.shape[0]
            #dump_cluster_index = clusters.shape[0]

            #new_clusters = numpy.empty( ( clusters.shape[0] + 1, clusters.shape[1] ) )
            #new_clusters[ : -1 ] = clusters
            #new_clusters[ -1 ] = numpy.nan
            #clusters = new_clusters

            print 'discarded %d cells' % ( numpy.sum( mask ) )
    
            print clusters.shape
    
            clusters = cluster.compute_centroids_from_partition( norm_features, partition )
    
            mask = partition >= 0
            clusterContainer = {
                'points' : norm_features[ mask ].copy(),
                'partition' : partition[ mask ].copy(),
                'clusters' : clusters.copy(),
                'dump_cluster_index' : dump_cluster_index,
                'mask' : mask.copy(),
                'featureNames' : featureNames,
                'Z' : Z,
                'method' : method,
                'featureset_index' : featureset_index,
                'exp_factor' : exp_factor,
                'param1' : param1,
                'param2': param2,
                'param3': param3,
                'param4': param4,
                'valid_feature_mask' : valid_feature_mask,
                'newFeatureIds' : newFeatureIds,
                'best_feature_mask' : best_feature_mask
            }

            partition = self.__partition_along_clusters( norm_features, clusterContainer, param3 )

            #print 'found %d non-discarded clusters' % clusters.shape[0]
            #for i in xrange( clusters.shape[0] ):
            #    cluster_size = numpy.sum( partition[:] == i )
            #    #if cluster_size < dissolve_cluster_threshold:
            #    print 'cluster %d: %d' % ( i, cluster_size )
    
    
            # Compute standard-deviation vector for each cluster
            stddevs = numpy.empty( clusters.shape )
            for i in xrange( clusters.shape[ 0 ] ):
                partition_mask = partition[:] == i
                stddevs[ i ] = numpy.std( norm_features[ partition_mask ], axis=0 )
    
            # Merge clusters.
            # Compute the distance matrix of the clusters.
    
            if method == 'seed':
    
                n = 0
                while clusters.shape[0] > param2:
        
                    dist_m = distance.minkowski_cdist( clusters, clusters, minkowski_p )
                    mask = numpy.identity( dist_m.shape[0], dtype=bool )
                    dist_m[ mask ] = numpy.inf
                    dist_m_min = numpy.min( dist_m, axis=1 )
                    dist_m_arg_min = numpy.argmin( dist_m, axis=1 )
                    dist_m_sorted = numpy.sort( dist_m_min, axis=0 )
                    dist_m_arg_sorted = numpy.argsort( dist_m_min, axis=0 )
        
                    dist = dist_m_sorted[ n ]
                    i = dist_m_arg_sorted[ n ]
                    j = dist_m_arg_min[ i ]
                    print 'merging %d and %d: dist=%f' % ( i, j, dist )
                    partition, clusters, weights =  self.__merge_clusters(
                            i,
                            j,
                            norm_features,
                            partition,
                            clusters,
                            weights
                    )
    
    
            """# Merge clusters whose distance is in the order of their standard deviations.
            # We compare the cluster distance to the sum of their projected standard
            # deviations multiplied by the parameter merge_cluster_factor plus the
            # parameter merge_cluster_offset
            # Compute the distance matrix of the clusters
            dist_m = distance.minkowski_cdist( clusters, clusters, minkowski_p )
            q =[]
            n = 0
            all_clusters_merged = False
            while not all_clusters_merged:
                all_clusters_merged = True
                for i in xrange( clusters.shape[ 0 ] ):
                    for j in xrange( clusters.shape[ 0 ] ):
                        if i != j:
                            # compute the normalized distance vector of the clusters
                            dist = dist_m[ i, j ]
                            dvec = clusters[i] - clusters[j]
                            dvec_norm = dvec / dist
                            #if dist != numpy.sqrt( numpy.sum( dvec ** 2 ) ):
                            #    print 'dist=%f, |dvec|=%f' % ( dist, numpy.sqrt( numpy.sum( dvec ** 2 ) ) )
                            # compute projected standard deviations
                            proj_stddev1 = abs( numpy.sum( dvec_norm * stddevs[i] ) )
                            proj_stddev2 = abs( numpy.sum( dvec_norm * stddevs[j] ) )
                            proj_stddev = min( proj_stddev1, proj_stddev2 )
                            # compute norm-value of the projected standard deviations
                            # compute distance threshold for merging
                            dist_threshold = merge_cluster_offset \
                              + merge_cluster_factor * proj_stddev
                            # check if the clusters should be merged
                            if dist < dist_threshold:
                                print 'merging cluster %d and %d' % ( i+n, j+n )
                                print 'dist=%f, threshold=%f, factor=%f, offset=%f' % ( dist, dist_threshold, merge_cluster_factor, merge_cluster_offset )
                                partition, clusters, weights = self.__merge_clusters(
                                        i,
                                        j,
                                        norm_features,
                                        partition,
                                        clusters,
                                        weights
                                )
                                all_clusters_merged = False
                                n += 1
                                break
                            else:
                                q.append( dist/dist_threshold )
                                #print 'not merging cluster %d and %d, dist/threshold = %f' % ( i+n, j+n, dist/dist_threshold )
                    if not all_clusters_merged:
                        break
    
            q = numpy.array( q )
            print numpy.mean( q ), numpy.median( q ), numpy.std( q )
            print 'merged %d clusters' % ( n )"""
    
    
            # Dissolve clusters which are smaller than dissolve_cluster_threshold
            # and keep doing this until all such clusters have been dissolved.
            # The cells of a dissolved cluster will be added to the cluster whose
            # centroid is nearest.
            n = 0
            all_clusters_dissolved = False
            while not all_clusters_dissolved:
                all_clusters_dissolved = True
                for i in xrange( clusters.shape[ 0 ] ):
                    # determine the mask of this cluster...
                    partition_mask = partition[:] == i
                    # and it's size...
                    cluster_size = numpy.sum( partition_mask )
                    # and check if it should be dissolved
                    if cluster_size < dissolve_cluster_threshold:
                        print 'dissolving cluster %d' % (i + n)
                        partition, clusters, weights = self.__dissolve_cluster( i, norm_features, partition, clusters, weights )
                        all_clusters_dissolved = False
                        n += 1
                        break
    
            """weights = numpy.empty( clusters.shape )
    
            all_stddevs = numpy.std( norm_features, axis=0 )
    
            # update feature weights for each cluster
            for k in xrange( clusters.shape[0] ):
    
                p_mask = partition[:] == k
                number_of_points = numpy.sum( p_mask )
    
                stddevs = numpy.std( norm_features[ p_mask ], axis=0)
    
                mask = stddevs[:] > 0
                inv_mask = numpy.invert( mask )
    
                weights[ k ][ mask ] = 1 - stddevs[ mask ] / all_stddevs[ mask ]
                weights[ k ][ inv_mask ] = 1.0
    
                min_weight = numpy.min( weights[ k ][ mask ] )
                weights[ k ][ mask ] = weights[ k ][ mask ] - min_weight
    
                max_weight = numpy.max( weights[ k ][ mask ] )
                weights[ k ][ mask ] = weights[ k ][ mask ]/ max_weight
    
                mask = numpy.logical_or( weights[ k ] > 1.0, weights[ k ] < 0.0 )
                if numpy.sum( mask ) > 0:
                    for i in xrange( mask.shape[0] ):
                        print 'i = %d' % i
                    print 'mask > 1.0: %d, k=%d' % ( numpy.sum( mask ), k )"""
    
    
            """print numpy.min( weights, axis=1 )
            print numpy.max( weights, axis=1 )
    
            for k in xrange( weights.shape[0] ):
                for i in xrange( weights.shape[1] ):
                    if weights[k,i] > 0.6:
                        print '%s=%f for cluster %d' % ( featureNames[i], weights[k,i], k )"""
    
            #weights_sum = numpy.sum( weights, axis=1 )
            #weights = weights.transpose() * weights.shape[1] / weights_sum
            #weights = weights.transpose()
    
            #print 'list of features used for clustering:'
            #for i in xrange( len( featureNames ) ):
            #    print '%d -> %s' % ( i, featureNames[i] )
    
            #weights = numpy.ones( clusters.shape )
    
            # print out some info about the dissolving
            print 'found %d non-dissolved clusters' % clusters.shape[0]
            sum = 0
            for i in xrange( clusters.shape[0] ):
                count = numpy.sum( partition[:] == i )
                sum += count
                print 'cluster %d: %d' % ( i, count )
    
            print 'clustered %d out of %d cells' % ( sum, norm_features.shape[0] )
    
            """for k in xrange( clusters.shape[ 0 ] ):
                # determine the mask of this cluster...
                partition_mask = partition[:] == k
                # and it's size...
                cluster_size = numpy.sum( partition_mask )
                # and check if it should be dissolved
                if cluster_size < dissolve_cluster_threshold:
                    # partition the cells of this cluster along
                    # the remaining clusters
                    new_partition = cluster.find_nearest_cluster(
                                                norm_features[ partition_mask ],
                                                clusters,
                                                weights
                    )
                    # update partitioning
                    partition[ partition_mask ] = new_partition"""

        else:

            mask = partition >= 0
            clusterContainer = {
                'points' : norm_features[ mask ].copy(),
                'partition' : partition[ mask ].copy(),
                'clusters' : clusters.copy(),
                'dump_cluster_index' : dump_cluster_index,
                'mask' : mask.copy(),
                'featureNames' : featureNames,
                'Z' : Z,
                'method' : method,
                'featureset_index' : featureset_index,
                'exp_factor' : exp_factor,
                'param1' : param1,
                'param2': param2,
                'param3': param3,
                'param4': param4,
                'valid_feature_mask' : valid_feature_mask,
                'newFeatureIds' : newFeatureIds,
                'best_feature_mask' : best_feature_mask
            }

        # compute the intra-cluster distances of all the samples to the centroid
        # of their corresponding cluster
        #intra_cluster_distances = cluster.compute_intra_cluster_distances(
        #    partition,
        #    norm_features,
        #    clusters,
        #    weights,
        #    minkowski_p
        #)
        intra_cluster_distances = None

        # compute the pairwise distance between all the clusters.
        # as a metric we use the mean of the distance of all the samples
        # in one cluster to the centroid of the other cluster.
        # this is not a real metric because it is NOT symmetric!
        #inter_cluster_distances = cluster.compute_inter_cluster_distances(
        #    partition,
        #    norm_features,
        #    clusters,
        #    weights,
        #    minkowski_p
        #)
        inter_cluster_distances = None

        oldPartition = partition
        partition = -numpy.ones(valid_cell_mask.shape, dtype=int)
        partition[valid_cell_mask] = oldPartition

        if not self.update_progress( 100 ):
            return False

        # keep the clusters, the partition, the silhouette and
        # the intra-cluster distances as public members
        self.clusterCellMask = cellMask
        self.clusterNormFeatures = norm_features
        self.clusterFeatures = features
        self.clusterFeatureIds = featureIds
        #self.clusterFeatureNames = featureNames
        self.clusters = clusters
        self.partition = partition
        #self.nonControlWeights = weights
        #self.nonControlSilhouette = silhouette
        self.intraClusterDistances = intra_cluster_distances
        self.interClusterDistances = inter_cluster_distances

        self.clusterDist = numpy.empty( ( self.pdc.objFeatures.shape[0], clusters.shape[0] ) )
        self.clusterDist[:,:] = numpy.inf
        print 'clusterDist.shape:', self.clusterDist.shape
        print 'cellMask.shape:', cellMask.shape
        self.clusterDist[cellMask] = distance.minkowski_cdist( norm_features, clusters )

        self.clusterContainer = clusterContainer

        # update state of the pipeline
        self.__state = self.PIPELINE_STATE_CLUSTERING

        # add the cluster ID to the object features
        tmpFeatures = self.pdc.objFeatures
        self.pdc.objFeatures = numpy.empty( ( tmpFeatures.shape[0], tmpFeatures.shape[1] + 1 ) )
        self.pdc.objFeatures[:,:-1] = tmpFeatures[:,:]
        cluster_ids = -numpy.ones( ( tmpFeatures.shape[0], ) )
        cluster_ids[cellMask] = self.partition
        self.pdc.objFeatures[:,-1] = cluster_ids[:]
        self.pdc.objFeatureIds[ 'ClusterID' ] = tmpFeatures.shape[1]
        del tmpFeatures

        return True

    @classmethod
    def get_hcluster_methods(cls):
        return [
            ('CLink', 'complete'),
            ('SLink', 'single'),
            ('ALink', 'average'),
            ('Ward', 'ward')
        ]

    def start_cluster_profiling(self, cluster_method, hcluster_method='average',
                                exp_factor=-1, profile_metric='quadratic_chi',
                                group_descriptions=['treatment','replicate'], progressCallback=None):
        """Start the pipeline thread to perform the cluster profiling
    
        Input parameters:
            - progressCallback: A thread-safe callback method. This method must
              accept an integer parameter that contains the progress of the
              pipeline thread (usually between 0 and 100)
        """

        self.progressCallback = progressCallback

        # run the quality-control within the thread
        self.start_method(self.run_cluster_profiling, cluster_method, hcluster_method,
                          exp_factor, profile_metric, group_descriptions)

    def run_cluster_profiling(self, cluster_method, hcluster_method='average',
                              exp_factor=-1, profile_metric='quadratic_chi',
                              group_descriptions=['treatment','replicate']):
        """Perform the cluster profiling
    
        Input parameters:
        """

        if exp_factor < 0:
            print 'computing cluster profiles...'
        else:
            print 'computing smooth cluster profiles...'

        from ..core import debug
        debug.start_debugging()
        debug.set_break()

        binSimilarityMatrix = cluster_profiles.compute_cluster_similarity_matrix( self.clusters )

        clusterProfileLabelsDict = {}
        clusterProfilesDict = {}
        distanceHeatmapDict = {}
        similarityHeatmapDict = {}
        dendrogramDict = {}

        for group_description in group_descriptions:
            print 'group_description:', group_description
            groups = grouping.get_groups(group_description, self.pdc, self.clusterCellMask)
            names,masks = zip(*groups)

            clusterProfileLabels = names
            clusterProfiles = cluster_profiles.compute_cluster_profiles(
                masks, self.clusterNormFeatures, self.clusters, exp_factor)
            print 'clusterProfiles.shape:', clusterProfiles.shape

            normalizationFactor = 0.5
            distanceHeatmap = cluster_profiles.compute_treatment_distance_map(clusterProfiles, profile_metric, 0.0, binSimilarityMatrix, normalizationFactor)
            similarityHeatmap = cluster_profiles.compute_treatment_similarity_map(distanceHeatmap, profile_metric)
            print 'distanceHeatmap.nan:', numpy.sum(numpy.isnan(distanceHeatmap))

            clusterProfileLabelsDict[group_description] = clusterProfileLabels
            clusterProfilesDict[group_description] = clusterProfiles
            distanceHeatmapDict[group_description] = distanceHeatmap
            similarityHeatmapDict[group_description] = similarityHeatmap

            for i in xrange(distanceHeatmap.shape[0]):
                for j in xrange(i+1, distanceHeatmap.shape[1]):
                    print '%s->%s:' % (clusterProfileLabels[i], clusterProfileLabels[j]), distanceHeatmap[i,j]

            global has_hcluster
            if has_hcluster:
                cdm = distanceHeatmap.copy()
                mask1 = numpy.isfinite(cdm)[0]
                mask2 = numpy.isfinite(cdm)[:,0]
                valid_mask = numpy.logical_or(mask1, mask2)
                cdm = cdm[valid_mask][:,valid_mask]
                cdm[numpy.identity(cdm.shape[0], dtype=bool)] = 0.0
                cdm = hcluster.squareform(cdm)
                Z = hcluster.linkage(cdm, hcluster_method )
            else:
                Z = None
            dendrogramDict[group_description] = Z

        if cluster_method == 'mahalanobis':
            for group_description in group_descriptions:
                print 'Computing mahalanobis distance of %s groups...' % group_description
                distanceHeatmap = distanceHeatmapDict[group_description]
                similarityHeatmap = similarityHeatmapDict[group_description]
                distanceHeatmap = numpy.zeros(distanceHeatmap.shape)
                similarityHeatmap = numpy.zeros(similarityHeatmap.shape)
                groups = grouping.get_groups(group_description, self.pdc, self.clusterCellMask)
                names,masks = zip(*groups)
                for i,ref_mask in enumerate(masks):
                    if numpy.any(ref_mask):
                        ref_points = self.clusterFeatures[ref_mask]
                        for j,mask in enumerate(masks):
                            #points = norm_features[mask]
                            if numpy.any(mask):
                                points = self.clusterFeatures[mask]
                                mahal_dist = distance.mahalanobis_distance(ref_points, points, fraction = 0.8)
                                distanceHeatmap[i,j] = numpy.mean(mahal_dist)
                            else:
                                distanceHeatmap[i,j] = numpy.NaN
                    else:
                        distanceHeatmap[i,:] = numpy.NaN
                distanceHeatmapDict[group_description] = distanceHeatmap

        debug.suspend_debugging()

        self.clusterProfileLabelsDict = clusterProfileLabelsDict
        self.clusterProfilesDict = clusterProfilesDict
        self.distanceHeatmapDict = distanceHeatmapDict
        self.similarityHeatmapDict = similarityHeatmapDict
        self.dendrogramDict = dendrogramDict

        return True

    def __partition_along_clusters(self, points, clusterContainer, minkowski_p=2):
        """Partition the samples to the cluster with the minimum distance
        
        Input parameters:
            - points: A nxm numpy array of the samples to be distributed.
              Rows are samples and columns are features.
            - clusters: A kxm numpy array of the clusters along which the
              samples should be distributed. Rows are clusters and
              columns are features.
            - minkowski_p: The power p of the minkowski metric,
              e.g. 2 for euclidean, 1 for city-block
        Output parameter:
            A one-dimensional numpy array of length n. For each sample the index
            of the cluster with the nearest centroid is specified."""

        method = clusterContainer[ 'method' ]

        clusters = clusterContainer[ 'clusters' ]

        if method not in [ 'k-means', 'k-medians', 'k-medoids' ]:

            clustered_mask = clusterContainer[ 'mask' ]
            clustered_points = clusterContainer[ 'points' ]
            clustered_partition = clusterContainer[ 'partition' ]

            inv_mask = numpy.invert( clustered_mask )

            partition = numpy.empty( ( points.shape[0], ), dtype=int )
            partition[ clustered_mask ] = clustered_partition

            # partition to nearest cluster
            #
            # calculate the distance of all the samples to the k-th cluster centroid
            dist_m = distance.minkowski_cdist( clusters, points[ inv_mask ], minkowski_p )
            #
            # find the cluster with the nearest centroid for each sample
            partition[ inv_mask ] = numpy.argmin( dist_m, 0 )
    

            # partition by minimizing ESS
            #partition[ inv_mask ] = cluster.partition_along_clusters( points[ inv_mask ], clustered_points, clustered_partition, clusters )

            #partition = cluster.partition_along_clusters( points, clustered_points, clustered_partition, clusters )

        else:

            # calculate the distance of all the samples to the k-th cluster centroid
            dist_m = distance.minkowski_cdist( clusters, points, minkowski_p )
    
            # find the cluster with the nearest centroid for each sample
            partition = numpy.argmin( dist_m, 0 )
    
        # return a one-dimensional numpy array of length n. For each sample the index
        # of the cluster with the nearest centroid is specified
        return partition


    #def __compute_cluster_profile(self, points, clusters, exp_factor, minkowski_p=2):

        #if points.shape[0] == 0:
            #return numpy.zeros( ( clusters.shape[0], ) )

        ## calculate the distance of all the samples to the k-th cluster centroid.
        ## rows represent clusters, columns represent samples
        #dist_m = distance.minkowski_cdist( clusters, points, minkowski_p )

        ##print 'dist_m:', dist_m[ :, 0 ]

        #"""max = numpy.max( dist_m, axis=0 )

        #sim_m = max - dist_m

        #sim_m = sim_m / numpy.float_( numpy.sum( sim_m, axis=0 ) )

        #sim_m = ( numpy.exp( 2 * sim_m ) - 1 ) / ( numpy.exp( 2 ) - 1 )

        #sim_m = sim_m / numpy.float_( numpy.sum( sim_m, axis=0 ) )

        #print 'sim_m:', sim_m[ : , 0 ]

        #profile = numpy.sum( sim_m, axis=1 )

        #print 'sum( profile ):', numpy.sum( profile ), 'sum( sim_m ):', numpy.sum( sim_m ), 'points.shape[0]:', points.shape[0]"""

        #"""min_i = numpy.argmin( dist_m, axis=0 )

        #profile = numpy.empty( ( clusters.shape[0], ) )

        #for i in xrange( clusters.shape[0] ):
            #mask = min_i == i
            #profile[ i ] = numpy.sum( mask )

        #print 'sum( profile ):', numpy.sum( profile ), 'points.shape[0]:', points.shape[0]"""


        #if exp_factor < 0:
            #argmin = numpy.argmin( dist_m, axis=0 )
            #profile = numpy.zeros( ( clusters.shape[0], ) )
            #for i in xrange( clusters.shape[0] ):
                #profile[ i ] += numpy.sum( argmin == i )
            #return profile


        #sim_m = 1.0 / dist_m

        ##print 'sim_m:', sim_m[ : , 0 ]

        #mask = numpy.isinf( sim_m )
        #col_mask = numpy.any( mask, axis=0 )

        #sim_m[ :, col_mask ] = 0.0
        #sim_m[ mask ] = 1.0

        ##print 'sum( sim_m ):', numpy.sum( sim_m )

        #if exp_factor > 0:

            #sim_m = sim_m / numpy.max( sim_m )

            #sim_m = ( numpy.exp( exp_factor * sim_m ) - 1 ) / ( numpy.exp( exp_factor ) - 1 )

        ## normalize the similarity matrix
        #sim_m = sim_m / numpy.sum( sim_m, axis=0 )

        ##print 'sim_m:', sim_m[ : , 0 ]

        ##mask = numpy.isnan( sim_m, axis= )
        ##sim_m[ mask ] = 1.0

        #profile = numpy.sum( sim_m, axis=1 )

        #print 'sum( profile ):', numpy.sum( profile ), 'sum( sim_m ):', numpy.sum( sim_m ), 'points.shape[0]:', points.shape[0]

        #return profile

    #def compute_cluster_profile(self, points, clusters, exp_factor=10.0, minkowski_p=2):
        #return self.__compute_cluster_profile( points, clusters, exp_factor, minkowski_p )

    #def __compute_affinity_profiles(self, masks, points, clusters, minkowski_p):
        #affinity_profiles = numpy.zeros((len(masks), clusters.shape[0]))
        ## calculate the distance of all pairs of cluster centroids ...
        #cluster_dist_m = distance.minkowski_cdist(
                #clusters, clusters, minkowski_p)
        ## ... and find the minimum
        #min_cluster_dist = numpy.min(cluster_dist_m[numpy.invert(
                #numpy.identity(cluster_dist_m.shape[0], dtype=bool)
        #)])
        ## Calculate the distance of all the samples to the k-th cluster
        ## centroid. Rows represent samples, columns represent clusters.
        #dist_m = distance.minkowski_cdist(points, clusters, minkowski_p)
        ## Compute affinity profile for each point and normalize it.
        #profiles = numpy.exp(-dist_m/min_cluster_dist)
        #profiles = profiles / numpy.sum(profiles, axis=1)[:,numpy.newaxis]
        ## Compute the mean affinity profile for each set of points.
        #for i in xrange(len(masks)):
            #mask = masks[i]
            #p = points[mask]
            #affinity_profiles[i] = numpy.sum(profiles[mask], axis=0) / numpy.sum(mask)
        #return affinity_profiles

    #def compute_cluster_profiles(self, masks, points, clusters,
                                 #exp_factor=10.0, minkowski_p=2):
        #"""if exp_factor < 0.0:
            #return self.__compute_affinity_profiles(
                        #masks, points, clusters, minkowski_p)
        #else:"""
        #if True:
            #clusterProfiles = numpy.zeros((len(masks), clusters.shape[0]))
            #for i in xrange( len( masks ) ):
                #mask = masks[ i ]
                #p = points[ mask ]
                #clusterProfiles[ i ] = self.__compute_cluster_profile(
                        #p, clusters, exp_factor, minkowski_p)
            #return clusterProfiles

    #def __compute_cluster_profiles(self, cell_mask, points, clusters, partition, exp_factor=10.0, minkowski_p=2):

        #clusterProfiles = numpy.zeros( ( len( self.pdc.treatments ), clusters.shape[0] ) )

        #if exp_factor < 0:
            #print 'computing cluster profiles...'
        #else:
            #print 'computing smooth cluster profiles...'

        #print 'treatments:', len( self.pdc.treatments )
        #for tr in self.pdc.treatments:

            #trMask = self.pdc.objFeatures[ cell_mask ][ : , self.pdc.objTreatmentFeatureId ] == tr.index
            #print 'treatment %s: %d' % ( tr.name, numpy.sum( trMask ) )

            #if exp_factor < 0:

                #for i in xrange( clusters.shape[0] ):
                    #clusterProfiles[ tr.index, i ] = numpy.sum( partition[ trMask ] == i )

            #else:

                #clusterProfiles[ tr.index ] = self.__compute_cluster_profile( points[ trMask ], clusters, exp_factor, minkowski_p )

        #return clusterProfiles

    """"clusterProfiles = numpy.zeros( ( len( self.pdc.treatments ), clusters.shape[0] ) )
    
    for tr in self.pdc.treatments:
    
        trMask = self.pdc.objFeatures[ cell_mask ][ : , self.pdc.objTreatmentFeatureId ] == tr.index
        print 'treatment %s: %d' % ( tr.name, numpy.sum( trMask ) )
    
        for i in xrange( clusters.shape[0] ):
            clusterProfiles[ tr.index, i ] = numpy.sum( partition[ trMask ] == i )
    
    return clusterProfiles"""


    def compute_feature_importance(self, points, point_mask):
        """Calculate the feature importance of a sub-population of cells
    
        Input parameters:
            - points: A nxm numpy array of all the samples.
              Rows are samples and columns are features.
            - point_mask: A 1-dimensional numpy array of length n masking
              the sub-population of interest.
            - all_mads: The median-absolute-deviation of each feature along all samples
            - all_mask: A mask of features that have a mad > 0
        Output parameter:
            A 1-dimensional numpy array of length m containing the importance of each feature
            for the sub-population."""

        return cluster.compute_feature_importance( points, point_mask )

    def __dissolve_cluster(self, cluster_index, points, partition, clusters, weights):

        # remove the cluster centroid from the list of centroids
        centroid_mask = numpy.empty( ( clusters.shape[0], ), dtype=numpy.bool)
        centroid_mask[:] = True
        centroid_mask[ cluster_index ] = False
        clusters = clusters[ centroid_mask ]
        if weights != None and len( weights.shape ) > 1:
            weights = weights[ centroid_mask ]
        partition_mask = partition[:] == cluster_index
        # reassign partition numbers
        reassign_mask = partition[:] > cluster_index
        partition[ reassign_mask ] -= 1
        # partition the cells of this cluster along
        # the remaining clusters
        new_partition = cluster.find_nearest_cluster(
                                    points[ partition_mask ],
                                    clusters,
                                    weights,
                                    minkowski_p
        )
        # update partitioning
        partition[ partition_mask ] = new_partition

        return partition, clusters, weights


    def dissolve_cluster(self, cluster_index):

        clusters = self.clusters
        partition = self.partition
        weights = self.nonControlWeights

        partition,clusters,weights = self.__dissolve_cluster(
                    cluster_index,
                    self.nonControlNormFeatures,
                    partition,
                    clusters,
                    weights
        )

        # compute the intra-cluster distances of all the samples to the centroid
        # of their corresponding cluster
        intra_cluster_distances = cluster.compute_intra_cluster_distances(
            partition,
            self.nonControlNormFeatures,
            clusters,
            weights,
            minkowski_p
        )

        # compute the pairwise distance between all the clusters.
        # as a metric we use the mean of the distance of all the samples
        # in one cluster to the centroid of the other cluster.
        # this is not a real metric because it is NOT symmetric!
        inter_cluster_distances = cluster.compute_inter_cluster_distances(
            partition,
            self.nonControlNormFeatures,
            clusters,
            weights,
            minkowski_p
        )

        self.clusters = clusters
        self.partition = partition
        self.nonControlWeights = weights
        self.intraClusterDistances = intra_cluster_distances
        self.interClusterDistances = inter_cluster_distances


    def __merge_clusters(self, cluster_index1, cluster_index2, points, partition, clusters, weights):

        if cluster_index1 == cluster_index2:
            return partition, clusters, weights
        elif cluster_index1 > cluster_index2:
            tmp = cluster_index1
            cluster_index1 = cluster_index2
            cluster_index2 = tmp

        # now we have cluster_index2 > cluster_index1

        # remove the second cluster centroid from the list of centroids
        centroid_mask = numpy.empty( ( clusters.shape[0], ), dtype=numpy.bool)
        centroid_mask[:] = True
        centroid_mask[ cluster_index2 ] = False
        clusters = clusters[ centroid_mask ]
        if weights != None and len( weights.shape ) > 1:
            weights = weights[ centroid_mask ]
        partition_mask = numpy.logical_or(
            partition[:] == cluster_index1,
            partition[:] == cluster_index2
        )
        # reassign partition numbers
        reassign_mask = partition[:] > cluster_index2
        partition[ reassign_mask ] -= 1
        # merge the partitioning of the two clusters
        partition[ partition_mask ] = cluster_index1
        clusters[ cluster_index1 ] = cluster.compute_centroid( points[ partition_mask ] )

        return partition, clusters, weights

    def merge_clusters(self, cluster_index1, cluster_index2):

        clusters = self.clusters
        partition = self.partition
        weights = self.nonControlWeights

        partition,clusters,weights = self.__merge_clusters(
                    cluster_index1,
                    cluster_index2,
                    self.nonControlNormFeatures,
                    partition,
                    clusters,
                    weights
        )

        # compute the intra-cluster distances of all the samples to the centroid
        # of their corresponding cluster
        intra_cluster_distances = cluster.compute_intra_cluster_distances(
            partition,
            self.nonControlNormFeatures,
            clusters,
            weights,
            minkowski_p
        )

        # compute the pairwise distance between all the clusters.
        # as a metric we use the mean of the distance of all the samples
        # in one cluster to the centroid of the other cluster.
        # this is not a real metric because it is NOT symmetric!
        inter_cluster_distances = cluster.compute_inter_cluster_distances(
            partition,
            self.nonControlNormFeatures,
            clusters,
            weights,
            minkowski_p
        )

        self.clusters = clusters
        self.partition = partition
        self.nonControlWeights = weights
        self.intraClusterDistances = intra_cluster_distances
        self.interClusterDistances = inter_cluster_distances
