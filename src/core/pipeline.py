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
utils.register_parameter( __name__, 'dissolve_cluster_threshold', utils.PARAM_INT, 'Minimum size of clusters which will not be dissolved', 200, 1, 1000 )



class Pipeline(Thread):
    """The Pipeline class represents the computational workflow. It represents all the steps
    like  quality-control, pre-filtering and clustering as methods

    Public methods:
    Public members:
    """

    __pyqtSignals__ = ('updateProgress',)

    def __init__(self, pdc, clusterConfiguration):
        """Constructor for Pipeline
    
        Input parameters:
            - pdc: PhenoNice data container
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

        # this keeps the callback method for the progress of a step within the pipeline
        self.progressCallback = None

        # this keeps the result of a step within the pipeline (if it was run in a thread)
        self.result = None

    def __del__(self):
        """Destructor for Pipeline
        """

        # wait until the thread has finished
        self.wait()

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


        print 'doing quality control'

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

        # indicate success
        return True


    def start_pre_filtering(self, progressCallback=None):
        """Start the pipeline thread to perform the pre-filtering
    
        Input parameters:
            - progressCallback: A thread-safe callback method. This method must
              accept an integer parameter that contains the progress of the
              pipeline thread (usually between 0 and 100)
        """

        self.progressCallback = progressCallback

        # run the quality-control within the thread
        self.start_method( self.run_pre_filtering )

    def run_pre_filtering(self):
        """Perform pre-filtering
        """

        if not self.update_progress( 0 ):
            return False

        # cutoff control cells and select the features to be used
        controlCellMask, nonControlCellMask, featureIds = analyse.cutoff_control_cells( self.pdc, self.validImageMask, self.validCellMask )
        #controlCellMask, nonControlCellMask, featureIds = analyse.cutoff_control_cells( self.pdc, featureNames, validImageMask, validCellMask )

        if controlCellMask == None:
            return False

        # extract the features of the non-control-like cells
        nonControlFeatures = self.pdc.objFeatures[ nonControlCellMask ]

        if not self.update_progress( 100 ):
            return False

        # keep the cell masks and the features of the non-control-like cells as public members
        self.controlCellMask = controlCellMask
        self.nonControlCellMask = nonControlCellMask
        self.nonControlFeatures = nonControlFeatures

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

        self.update_progress( 0 )

        global progress
        progress = 0.0
        def cluster_callback(iterations, swaps):
            global progress
            progress += 2
            self.update_progress( int( progress + 0.5 ) )
            return True

        best_num_of_clusters, gaps, sk = cluster.determine_num_of_clusters( self.nonControlTransformedFeatures, max_num_of_clusters, num_of_reference_datasets )

        self.update_progress( 100 )

        return best_num_of_clusters, gaps, sk


    def start_clustering(self, supercluster_index=0, num_of_clusters=-1, calculate_silhouette=False, progressCallback=None):
        """Start the pipeline thread to perform the clustering
    
        Input parameters:
            - supercluster_index: The index of the supercluster to use
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
        self.start_method( self.run_clustering, supercluster_index, num_of_clusters, calculate_silhouette )

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


    def run_clustering(self, supercluster_index=0, num_of_clusters=-1, calculate_silhouette=False):
        """Perform the clustering
    
        Input parameters:
            - supercluster_index: The index of the supercluster to use
            - num_of_clusters: The number of clusters to be used.
              if the value -1 is passed, an educated guess is made
            - calculate_silhouette: Determins wether to calculate the silhouette
              of the final clustering
        """

        # If num_of_clusters is not provided, check wether the parameter
        # number_of_clusters has been defined.
        # Otherwise make an 'educated' guess
        if ( num_of_clusters == None ) or num_of_clusters <= 0:

            try:
                number_of_clusters
            except:
                number_of_clusters = len( self.pdc.treatments )

            num_of_clusters = number_of_clusters


        self.update_progress( 0 )

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
            self.update_progress( int( progress + 0.5 ) )
            return True

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
        norm_features = norm_features[ : , valid_mask ]
        # print out some info
        print 'number of features used for clustering: %d' % numpy.sum( valid_mask )
        print 'number of features not used for clustering: %d' % numpy.sum( invalid_mask )

        # Call the actual clustering-routine.
        # Use K-Means and cluster the non-control-features
        # (see cluster.cluster for details)
        #partition,clusters = cluster.cluster_by_dist( self.nonControlTransformedFeatures, num_of_clusters, clustering_swap_threshold, cluster_callback )
        partition,clusters,silhouette,weights = cluster.cluster(
                cluster.CLUSTER_METHOD_KMEANS,
                norm_features,
                num_of_clusters,
                minkowski_p,
                calculate_silhouette,
                cluster_callback
        )
        """partition,clusters = cluster.cluster_by_fermi_dirac_dist(
                                        nonControlTransformedFeatures,
                                        num_of_clusters,
                                        clustering_swap_threshold,
                                        fermi_dirac_sim_sharpness,
                                        cluster_callback
        )"""


        weights = cluster.compute_feature_weighting( partition, points, clusters, minkowski_p )


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
            sum += numpy.sum( partition[:] == i )
            print 'cluster %d: %d' % ( i, numpy.sum( partition[:] == i ) )

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

        # compute the intra-cluster distances of all the samples to the centroid
        # of their corresponding cluster
        intra_cluster_distances = cluster.compute_intra_cluster_distances(
            partition,
            norm_features,
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
            norm_features,
            clusters,
            weights,
            minkowski_p
        )

        self.update_progress( 100 )

        # keep the clusters, the partition, the silhouette and
        # the intra-cluster distances as public members
        self.nonControlNormFeatures = norm_features
        self.nonControlClusters = clusters
        self.nonControlPartition = partition
        self.nonControlWeights = weights
        self.nonControlSilhouette = silhouette
        self.nonControlIntraClusterDistances = intra_cluster_distances
        self.nonControlInterClusterDistances = inter_cluster_distances
        self.nonControlFeatureIds = featureIds
        self.nonControlFeatureNames = featureNames

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
            return
        elif cluster_index1 > cluster_index2:
            tmp = cluster_index1
            cluster_index1 = cluster_index2
            cluster_index2 = tmp

        # remove the cluster centroid from the list of centroids
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

