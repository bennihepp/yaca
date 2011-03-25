""" This module contains the Pipeline class. This class defines the whole computational workflow
    of the application.
    The command-line interface or the graphical user-interface only need to know about this class.
    All the steps like quality-control, pre-filtering and clustering are represented
    as methods in the Pipeline class."""

import sys
import os

import numpy

import analyse, distance, quality_control, cluster


from thread_utils import Thread

from thread_utils import Thread,SIGNAL



# define necessary parameters for this module (see parameter_utils.py for details)
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
        'nonControlClusters',
        'nonControlPartition',
        'nonControlSilhouette',
        'nonControlWeights',
        'nonControlIntraClusterDistances',
        'nonControlInterClusterDistances',
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

        # initialize some public members
        # the mask of images and cells that passed quality control
        self.validImageMask = None
        self.validCellMask = None
        # the mask of control-like and not-control-like cells (that passed quality control)
        self.controlCellMask = None
        self.nonControlCellMask = None
        # ?? TODO ??
        self.featureIds = None
        self.featureNames = None
        # the features of the not-control-like cells
        self.nonControlFeatures = None
        # the features of the not-control-like cells used for clustering
        self.nonControlClusterFeatures = None
        # the clusters of the non-control-like cells that were found
        self.nonControlClusters = None
        # the partition of the non-control-like cells that was found
        self.nonControlPartition = None
        # ?? TODO ?? the silhouette of the non-control-like cells that was found
        self.nonControlSilhouette = None
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

            clusters = clusterContainer[ 'clusters' ]    
            featureNames = clusterContainer[ 'featureNames' ]
            Z = clusterContainer[ 'Z' ]
            tree = Z
            method = clusterContainer[ 'method' ]

            # determine the IDs of the features to be used
            featureIds = []
            for featureName in featureNames:
                featureIds.append( self.pdc.objFeatureIds[ featureName ] )

            id_mapping = numpy.array( featureIds )
    
            # extract the features for the clustering
            features = self.pdc.objFeatures[ self.nonControlCellMask ][ : , featureIds ]
    
            norm_features, valid_mask = self.__compute_normalized_features( featureNames )

            if method != 'kd-tree':

                partition = self.__partition_along_clusters( norm_features, clusterContainer )
                #partition = self.__partition_along_clusters( norm_features, clusters )

            else:

                partition = cluster.partition_along_kd_tree( tree, norm_features )

            clusterProfiles = self.__compute_cluster_profiles( self.nonControlCellMask, norm_features, clusters, partition, exp_factor )
    
            mask = partition >= 0
            clusterContainer = {
                'points' : norm_features[ mask ].copy(),
                'partition' : partition[ mask ].copy(),
                'clusters' : clusters.copy(),
                'method' : method,
                'mask' : mask.copy(),
                'featureNames' : featureNames,
                'Z' : Z
            }

            # keep the clusters, the partition, the silhouette and
            # the intra-cluster distances as public members
            self.nonControlNormFeatures = norm_features
            self.nonControlClusters = clusters
            self.nonControlPartition = partition
            self.nonControlWeights = None
            self.nonControlIntraClusterDistances = None
            self.nonControlInterClusterDistances = None
            self.nonControlFeatureIds = featureIds
            self.nonControlFeatureNames = featureNames
            self.nonControlClusterProfiles = clusterProfiles
            self.Z = Z
            self.method = method
            self.clusterContainer = clusterContainer

        finally:

            if filename != None:
                file.close()

    def save_clusters(self, file):

        #dict = { 'clusters' : self.nonControlClusters, 'featureNames' : self.nonControlFeatureNames }
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

        # indicate success
        return True


    def start_pre_filtering(self, controlFilterMode=analyse.FILTER_MODE_MEDIAN_xMAD_AND_LIMIT, progressCallback=None):
        """Start the pipeline thread to perform the pre-filtering
    
        Input parameters:
            - progressCallback: A thread-safe callback method. This method must
              accept an integer parameter that contains the progress of the
              pipeline thread (usually between 0 and 100)
        """

        self.progressCallback = progressCallback

        # run the quality-control within the thread
        self.start_method( self.run_pre_filtering, controlFilterMode )


    def run_pre_filtering(self, controlFilterMode=analyse.FILTER_MODE_MEDIAN_xMAD):
        """Perform pre-filtering
        """

        if not self.update_progress( 0 ):
            return False

        # reject cells which have NaN or Inf values
        validCellMask, featureIds, bad_feature_cell_count = analyse.reject_cells_with_bad_features( self.pdc, self.validImageMask, self.validCellMask )

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


        # reject control cells and select the features to be used
        controlCellMask, nonControlCellMask, featureIds, mahal_dist, cell_selection_stats = analyse.reject_control_cells( self.pdc, self.validImageMask, self.validCellMask, controlFilterMode)
        #controlCellMask, nonControlCellMask, featureIds = analyse.cutoff_control_cells( self.pdc, featureNames, validImageMask, validCellMask )

        self.cell_selection_stats = cell_selection_stats

        if controlCellMask == None:
            return False

        # add the mahalanobis distance to the object features
        tmpFeatures = self.pdc.objFeatures
        self.pdc.objFeatures = numpy.empty( ( tmpFeatures.shape[0], tmpFeatures.shape[1] + 1 ) )
        self.pdc.objFeatures[:,:-1] = tmpFeatures[:,:]
        self.pdc.objFeatures[:,-1] = numpy.sqrt( mahal_dist[:] )
        self.pdc.objFeatureIds[ 'Mahalanobis Distance' ] = tmpFeatures.shape[1]
        del tmpFeatures

        # reject manually selected treatments
        for tr_name in reject_treatments:
            tr_id = self.pdc.treatmentByName[ tr_name ]
            tr = self.pdc.treatments[ tr_id ]
            tr_mask = self.pdc.objFeatures[ :, self.pdc.objTreatmentFeatureId ] == tr.rowId
            tmp_mask = numpy.logical_and( nonControlCellMask, tr_mask )
            controlCellMask[ tmp_mask ] = True
            nonControlCellMask[ tr_mask ] = False

        # create a list of treatments which are not in 'reject-treatments'
        validTreatments = []
        validTreatmentIds = []
        for tr in self.pdc.treatments:
            if tr.name not in reject_treatments:
                validTreatments.append( tr )
                validTreatmentIds.append( tr.rowId )

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

        # keep the list of valid treatments
        self.validTreatments = validTreatments
        self.validTreatmentIds = validTreatmentIds

        # update state of the pipeline
        self.__state = self.PIPELINE_STATE_PRE_FILTERING

        # indicate success
        return True


    #def run_pipeline(self, supercluster_index=0, progressCallback=None):
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
    #    featureNames = self.clusterConfiguration[ supercluster_index ][ 1 ]
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
    #    self.nonControlClusters = clusters
    #    self.nonControlPartition = partition"""
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


    def start_clustering(self, method, index=0, param1=-1, param2=-1, param3=-1, param4=-1, exp_factor=-1, calculate_silhouette=False, progressCallback=None):
        """Start the pipeline thread to perform the clustering
    
        Input parameters:
            - index: The index of the supercluster to use
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
        self.start_method( self.run_clustering, method, index, param1, param2, param3, param4, exp_factor, calculate_silhouette )

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


    def __compute_normalized_features(self, featureNames):

        featureIds = []
        for featureName in featureNames:
            featureIds.append( self.pdc.objFeatureIds[ featureName ] )

        id_mapping = numpy.array( featureIds )

        # extract the features for the clustering
        features = self.pdc.objFeatures[ self.nonControlCellMask ][ : , featureIds ]

        # normalize the features
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
        valid_mask = numpy.logical_not( invalid_mask )

        # only keep valid features
        norm_features = norm_features[ : , valid_mask ]

        return norm_features, valid_mask

    def compute_normalized_features(self, supercluster_index):

        # determine the IDs of the features to be used
        featureNames = self.clusterConfiguration[ supercluster_index ][ 1 ]
        return self.__compute_normalized_features( featureNames )


    def run_clustering(self, method, supercluster_index=0, param1=-1, param2=-1, param3=-1, param4=-1, exp_factor=-1, calculate_silhouette=False):
        """Perform the clustering
    
        Input parameters:
            - method: the clustering method to use (see cluster.get_hcluster_methods)
            - supercluster_index: The index of the supercluster to use
            - param1: first parameter for the clustering (see cluster.hcluster)
            - param2: second parameter for the clustering (see cluster.hcluster)
            - calculate_silhouette: Determins wether to calculate the silhouette
              of the final clustering
        """

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
        featureNames = self.clusterConfiguration[ supercluster_index ][ 1 ]
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
        #norm_features = norm_features[ : , valid_mask ]

        norm_features, valid_mask = self.compute_normalized_features( supercluster_index )

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

        #print 'importing time module...'
        #import time
        #print 'imported time module'

        #t1 = time.time()
        #c1 = time.clock()
        #print 't1: %.2f' % t1
        #print 'c1: %.2f' % c1

        #partition,clusters,Z = cluster.cluster_hierarchical_seeds( norm_features, num_of_clusters, objects_to_cluster, minkowski_p )
        #dist_threshold = 15.0
        #partition,clusters,Z = cluster.hcluster( method, norm_features, param1, param2, supercluster_index, param3 )
        partition,clusters,Z = cluster.hcluster_special( method, norm_features, param1, param2, param4, supercluster_index, param3 )
        self.Z = Z

        #c2 = time.clock()
        #t2 = time.time()
        #print 't2: %.2f' % t2
        #print 'c2: %.2f' % c2

        #dc = c2 - c1
        #dt = t2 - t1
        #print 'clocks: %.2f' % dc
        #print 'time: %.2f' % dt

        print partition.shape
        if method not in [ 'kd-tree', 'random' ]:
            print clusters.shape
            print Z.shape

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
            mapping = identity_arr[ valid_mask ]
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

        if False:

            f = open('/home/hepp/clusters.xls','w')

            row = []
            row.append( '' )
            row.append( '' )
            identity_arr = numpy.arange( features.shape[1] )
            mapping = identity_arr[ valid_mask ]
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
                feat = features[:,valid_mask][p_mask]
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
            mapping = identity_arr[ valid_mask ]
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
                feat = features[:,valid_mask][p_mask]
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
            mapping = identity_arr[ valid_mask ]
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
                feat = features[:,valid_mask][p_mask]
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
            mapping = identity_arr[ valid_mask ]
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
                feat = features[:,valid_mask][p_mask]
                nfeat = norm_features[p_mask]
                for i in xrange( nfeat.shape[0] ):
                    row = []
                    row.append( '%d' % i )
                    row.append( '%d' % k )
                    for j in xrange( nfeat.shape[1] ):
                        row.append( '%f' % ( nfeat[ i, j ] ) )
                    f.write( ','.join( row ) + '\n' )
            f.close()


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


        if method not in [ 'kd-tree', 'random' ]:

            #if method == 'ward':
            discard_cluster_threshold = param4
    
            print 'threshold=', discard_cluster_threshold
    
    
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
    
            print 'discarded %d cells' % ( numpy.sum( mask ) )
    
            #new_clusters = numpy.empty( ( clusters.shape[0] + 1, clusters.shape[1] ) )
            #new_clusters[ : -1 ] = clusters
            #new_clusters[ -1 ] = numpy.nan
            #clusters = new_clusters
    
            print clusters.shape
    
            clusters = cluster.compute_centroids_from_partition( norm_features, partition )
    
            mask = partition >= 0
            clusterContainer = {
                'points' : norm_features[ mask ].copy(),
                'partition' : partition[ mask ].copy(),
                'clusters' : clusters.copy(),
                'method' : method,
                'mask' : mask.copy(),
                'featureNames' : featureNames,
                'Z' : Z
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
    
            print 'clustered %d cells' % sum
    
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
                'method' : method,
                'mask' : mask.copy(),
                'featureNames' : featureNames,
                'Z' : Z
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


        clusterProfiles = self.__compute_cluster_profiles( self.nonControlCellMask, norm_features, clusters, partition,  exp_factor )

        if not self.update_progress( 100 ):
            return False

        # keep the clusters, the partition, the silhouette and
        # the intra-cluster distances as public members
        self.nonControlNormFeatures = norm_features
        self.nonControlClusters = clusters
        self.nonControlPartition = partition
        self.nonControlWeights = weights
        #self.nonControlSilhouette = silhouette
        self.nonControlIntraClusterDistances = intra_cluster_distances
        self.nonControlInterClusterDistances = inter_cluster_distances
        self.nonControlFeatureIds = featureIds
        self.nonControlFeatureNames = featureNames
        self.nonControlClusterProfiles = clusterProfiles

        self.clusterContainer = clusterContainer

        # update state of the pipeline
        self.__state = self.PIPELINE_STATE_CLUSTERING

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

            partition[ inv_mask ] = cluster.partition_along_clusters( points[ inv_mask ], clustered_points, clustered_partition, clusters )

            #partition = cluster.partition_along_clusters( points, clustered_points, clustered_partition, clusters )

        else:

            # calculate the distance of all the samples to the k-th cluster centroid
            dist_m = distance.minkowski_cdist( clusters, points, minkowski_p )
    
            # find the cluster with the nearest centroid for each sample
            partition = numpy.argmin( dist_m, 0 )
    
        # return a one-dimensional numpy array of length n. For each sample the index
        # of the cluster with the nearest centroid is specified
        return partition


    def __compute_cluster_profile(self, points, clusters, exp_factor, minkowski_p=2):

        if points.shape[0] == 0:
            return numpy.zeros( ( clusters.shape[0], ) )

        # calculate the distance of all the samples to the k-th cluster centroid.
        # rows represent clusters, columns represent samples
        dist_m = distance.minkowski_cdist( clusters, points, minkowski_p )

        #print 'dist_m:', dist_m[ :, 0 ]

        """max = numpy.max( dist_m, axis=0 )

        sim_m = max - dist_m

        sim_m = sim_m / numpy.float_( numpy.sum( sim_m, axis=0 ) )

        sim_m = ( numpy.exp( 2 * sim_m ) - 1 ) / ( numpy.exp( 2 ) - 1 )

        sim_m = sim_m / numpy.float_( numpy.sum( sim_m, axis=0 ) )

        print 'sim_m:', sim_m[ : , 0 ]

        profile = numpy.sum( sim_m, axis=1 )

        print 'sum( profile ):', numpy.sum( profile ), 'sum( sim_m ):', numpy.sum( sim_m ), 'points.shape[0]:', points.shape[0]"""

        """min_i = numpy.argmin( dist_m, axis=0 )

        profile = numpy.empty( ( clusters.shape[0], ) )

        for i in xrange( clusters.shape[0] ):
            mask = min_i == i
            profile[ i ] = numpy.sum( mask )

        print 'sum( profile ):', numpy.sum( profile ), 'points.shape[0]:', points.shape[0]"""

        sim_m = 1.0 / dist_m

        #print 'sim_m:', sim_m[ : , 0 ]

        mask = numpy.isinf( sim_m )
        col_mask = numpy.any( mask, axis=0 )

        sim_m[ :, col_mask ] = 0.0
        sim_m[ mask ] = 1.0

        #print 'sum( sim_m ):', numpy.sum( sim_m )

        sim_m = sim_m / numpy.max( sim_m )

        sim_m = ( numpy.exp( exp_factor * sim_m ) - 1 ) / ( numpy.exp( exp_factor ) - 1 )

        # normalize the similarity matrix
        sim_m = sim_m / numpy.sum( sim_m, axis=0 )

        #print 'sim_m:', sim_m[ : , 0 ]

        #mask = numpy.isnan( sim_m, axis= )
        #sim_m[ mask ] = 1.0

        profile = numpy.sum( sim_m, axis=1 )

        print 'sum( profile ):', numpy.sum( profile ), 'sum( sim_m ):', numpy.sum( sim_m ), 'points.shape[0]:', points.shape[0]

        return profile

    def __compute_cluster_profiles(self, cell_mask, points, clusters, partition,  exp_factor=10.0):

        clusterProfiles = numpy.zeros( ( len( self.pdc.treatments ), clusters.shape[0] ) )

        if exp_factor > 0:
            print 'computing smooth cluster profiles...'
        else:
            print 'computing cluster profiles...'

        for tr in self.pdc.treatments:

            trMask = self.pdc.objFeatures[ cell_mask ][ : , self.pdc.objTreatmentFeatureId ] == tr.rowId
            print 'treatment %s: %d' % ( tr.name, numpy.sum( trMask ) )

            if exp_factor > 0:

                clusterProfiles[ tr.rowId ] = self.__compute_cluster_profile( points[ trMask ], clusters, exp_factor )

            else:

                for i in xrange( clusters.shape[0] ):
                    clusterProfiles[ tr.rowId, i ] = numpy.sum( partition[ trMask ] == i )

        return clusterProfiles

        """"clusterProfiles = numpy.zeros( ( len( self.pdc.treatments ), clusters.shape[0] ) )

        for tr in self.pdc.treatments:

            trMask = self.pdc.objFeatures[ cell_mask ][ : , self.pdc.objTreatmentFeatureId ] == tr.rowId
            print 'treatment %s: %d' % ( tr.name, numpy.sum( trMask ) )

            for i in xrange( clusters.shape[0] ):
                clusterProfiles[ tr.rowId, i ] = numpy.sum( partition[ trMask ] == i )

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

        clusters = self.nonControlClusters
        partition = self.nonControlPartition
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

        self.nonControlClusters = clusters
        self.nonControlPartition = partition
        self.nonControlWeights = weights
        self.nonControlIntraClusterDistances = intra_cluster_distances
        self.nonControlInterClusterDistances = inter_cluster_distances


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

        clusters = self.nonControlClusters
        partition = self.nonControlPartition
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

        self.nonControlClusters = clusters
        self.nonControlPartition = partition
        self.nonControlWeights = weights
        self.nonControlIntraClusterDistances = intra_cluster_distances
        self.nonControlInterClusterDistances = inter_cluster_distances

