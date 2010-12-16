import sys
import os

import analyse, distance, quality_control, cluster


import parameter_utils as utils

from thread_utils import Thread

from thread_utils import Thread,SIGNAL



__dict__ = sys.modules[ __name__ ].__dict__

utils.register_module( __name__, 'Image analysis pipeline', __dict__ )

utils.register_parameter( __name__, 'number_of_clusters', utils.PARAM_INT, 'Number of clusters (Default is number of treatment)', None, 1, None, True )

utils.register_parameter( __name__, 'clustering_swap_threshold', utils.PARAM_INT, 'Swap threshold for the clustering', 1, 1, None )

utils.register_parameter( __name__, 'fermi_dirac_sim_sharpness', utils.PARAM_FLOAT, 'Sharpness for the fermi-dirac-similarity', 2.0, 0.0, None )

utils.register_parameter( __name__, 'minkowski_p', utils.PARAM_INT, 'Parameter p for the Minkowski metric', 2, 1, None )



"""PipelineBaseClass = object
QtThreading = False

if 'PyQt4.QtCore' in sys.modules:
    from PyQt4.QtCore import *
    PipelineBaseClass = QThread
    QtThreading = True"""

class Pipeline(Thread):

    __pyqtSignals__ = ('updateProgress',)

    def __init__(self, adc, clusterConfiguration):

        print 'Pipeline init'
        Thread.__init__( self )

        self.adc = adc

        self.clusterConfiguration = clusterConfiguration

        self.validImageMask = None
        self.validCellMask = None

        self.controlCellMask = None
        self.nonControlCellMask = None

        self.featureIds = None
        self.featureNames = None

        self.nonControlFeatures = None
        self.nonControlTransformedFeatures = None

        self.nonControlClusters = None
        self.nonControlPartition = None
        self.nonControlSilhouette = None

        self.progressCallback = None

        self.result = None

    def __del__(self):
        self.wait()

    def stop(self):
        self.continue_running = False

    def update_progress(self, progress):
        if self.progressCallback != None:
            self.progressCallback( progress )
        else:
            self.emit( SIGNAL('updateProgress'), progress )

        return not self.has_been_stopped()

    def run(self):
        self.result = self.run_pipeline()

    def run_pipeline(self, supercluster_index=0, progressCallback=None):

        self.progressCallback = progressCallback

        if not self.update_progress( 0 ):
            return False

        print 'number of images: %d' % len( self.adc.images )
        print 'number of objects: %d' % len( self.adc.objects )


        if not self.update_progress( 20 ):
            return False


        print 'doing quality control'

        # do some quality control of the images and cells
        validImageMask, validCellMask = quality_control.quality_control( self.adc )


        if not self.update_progress( 70 ):
            return False


        #imgFeatures = adc.imgFeatures[ validImageMask ]
        #objFeatures = adc.objFeatures[ validCellMask ]

        print 'number of valid images: %d' % validImageMask.sum()
        print 'number of valid cells: %d' % validCellMask.sum()


        featureNames = self.clusterConfiguration[ supercluster_index ][ 1 ]

        # cutoff control cells and select the features to be used
        controlCellMask, nonControlCellMask, featureIds = analyse.cutoff_control_cells( self.adc, featureNames, validImageMask, validCellMask )

        if controlCellMask == None:
            return False

        if not self.update_progress( 80 ):
            return False


        featureNames = list( featureIds )
        for k,v in self.adc.objFeatureIds.iteritems():
            if v in featureNames:
                i = featureNames.index( v )
                if i >= 0:
                    featureNames[ i ] = k


        nonControlFeatures = self.adc.objFeatures[ nonControlCellMask ]

        # transform features into mahalanobis space and calculate the transformation matrix
        mahalTransformation = distance.mahalanobis_transformation( nonControlFeatures[ : , featureIds ] )

        # transformate data points into mahalanobis space
        nonControlTransformedFeatures = distance.transform_features( nonControlFeatures[ : , featureIds ], mahalTransformation )

        #print 'Using the following features:'
        #print '\n'.join(mahalFeatureNames)
        print 'using %d features' % len(featureIds)
    
        if not self.update_progress( 100 ):
            return False


        """self.update_progress( 50 )


        global progress
        progress = 50.0
        def cluster_callback(iterations, swaps):
            global progress
            progress += 0.5
            self.update_progress( int( progress + 0.5 ) )
            return True

        # cluster objects in mahalanobis space
        try:
            number_of_clusters
        except:
            number_of_clusters = len( self.adc.treatments )

        partition,clusters = cluster.cluster_by_dist( nonControlTransformedFeatures, number_of_clusters, clustering_swap_threshold, cluster_callback )

        #partition,clusters = cluster.cluster_by_fermi_dirac_dist(
        #                                nonControlTransformedFeatures,
        #                                number_of_clusters,
        #                                clustering_swap_threshold,
        #                                fermi_dirac_sim_sharpness,
        #                                cluster_callback
        #)


        self.update_progress( 100 )


        self.nonControlClusters = clusters
        self.nonControlPartition = partition"""

        self.validImageMask = validImageMask
        self.validCellMask = validCellMask

        self.controlCellMask = controlCellMask
        self.nonControlCellMask = nonControlCellMask

        self.featureIds = featureIds
        self.featureNames = featureNames

        self.nonControlFeatures = nonControlFeatures
        self.nonControlTransformedFeatures = nonControlTransformedFeatures

        return True


    def run_clustering(self, num_of_clusters=-1, progressCallback=None):

        if ( num_of_clusters == None ) or num_of_clusters <= 0:
            
            # cluster objects in mahalanobis space
            try:
                number_of_clusters
            except:
                number_of_clusters = len( self.adc.treatments )

            num_of_clusters = number_of_clusters


        self.progressCallback = progressCallback

        self.update_progress( 0 )

        global progress
        progress = 0.0
        def cluster_callback(iterations, swaps):
            global progress
            progress += 2
            self.update_progress( int( progress + 0.5 ) )
            return True

        #partition,clusters = cluster.cluster_by_dist( self.nonControlTransformedFeatures, num_of_clusters, clustering_swap_threshold, cluster_callback )
        partition,clusters,silhouette = cluster.cluster(
                cluster.CLUSTER_METHOD_KMEANS,
                self.nonControlTransformedFeatures,
                num_of_clusters,
                minkowski_p,
                cluster_callback
        )

        """partition,clusters = cluster.cluster_by_fermi_dirac_dist(
                                        nonControlTransformedFeatures,
                                        num_of_clusters,
                                        clustering_swap_threshold,
                                        fermi_dirac_sim_sharpness,
                                        cluster_callback
        )"""


        self.update_progress( 100 )


        self.nonControlClusters = clusters
        self.nonControlPartition = partition
        self.nonControlSilhouette = silhouette



