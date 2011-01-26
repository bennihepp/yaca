""" This module is used for preliminary analysation of the cells.
    Feature selection and the cutoff of control cells is done here."""

import sys
import os

import numpy
import scipy
import scipy.linalg

import distance


# define necessary parameters for this module (see parameter_utils.py for details)
#
import parameter_utils as utils
#
__dict__ = sys.modules[ __name__ ].__dict__
#
utils.register_module( __name__, 'Analysation of images', __dict__ )
#
import importer
utils.add_required_state( __name__, importer.__name__, 'imported' )
#
utils.register_parameter( __name__, 'control_treatment_names', utils.PARAM_TREATMENTS, 'Names of the control treatments' )
#
utils.register_parameter( __name__, 'mahal_dist_cutoff_fraction', utils.PARAM_FLOAT, 'Fraction of control cells to use for mahalanobis distance', 0.9, 0.0, 1.0 )
#
utils.register_parameter( __name__, 'min_stddev_threshold', utils.PARAM_FLOAT, 'Minimum threshold for standard deviation of features', 0.01, 0.0, None )
#
utils.register_parameter( __name__, 'max_correlation_threshold', utils.PARAM_FLOAT, 'Maximum threshold for feature correlation', 0.9, 0.0, 1.0 )
#
utils.register_parameter( __name__, 'control_cutoff_threshold', utils.PARAM_FLOAT, 'Threshold for cutoff of control-like cells', 1.0, 0.0, None )
#
utils.register_parameter( __name__, 'filter_obj_feature_id_names', utils.PARAM_OBJ_FEATURES, 'Features to use for pre-filtering', [] )



def cutoff_control_cells( pdc, imageMask, cellMask ):
    """Make a heuristic guess about which cells are control-like cells and which cells are not

    Input parameters:
      - pdc: PhenoNice data container
      - imageMask: Mask of images to take into account
      - cellMask: Mask of cells to take into account
    Output values (tuple):
      - Mask of suspected control cells
      - Mask of suspected non-control cells
      - List of feature IDs that have been taken into account
    """

    # trivial check
    # if filter_obj_feature_id_names is empty, assume that all cells are non-control like cells
    if len( filter_obj_feature_id_names ) == 0:
        emptyMask = numpy.empty( (pdc.objFeatures.shape[0],) )
        emptyMask[ : ] = False
        return emptyMask, cellMask, []

    # determine the feature IDs corresponding to the feature names in filter_obj_feature_id_names
    featureIds = []
    for featureName in filter_obj_feature_id_names:
        featureIds.append( pdc.objFeatureIds[ featureName ] )

    # TODO: this is just for testing purpose
    #return cellMask, cellMask, featureIds

    # extract the image- and object-features
    imgFeatures = pdc.imgFeatures[ imageMask ]
    objFeatures = pdc.objFeatures[ cellMask ]


    # create a mask that contains all the cells in every control treatment
    #
    # we start with an empty mask
    control_treatment_mask = numpy.empty( (pdc.objFeatures.shape[0],) )
    control_treatment_mask[ : ] = False
    # and for each control treatment...
    for name in control_treatment_names:
        control_treatment = pdc.treatments[ pdc.treatmentByName[ name ] ]
        # we add the corresponding cells to the mask
        control_treatment_mask = numpy.logical_or(
                control_treatment_mask,
                pdc.objFeatures[ : , pdc.objTreatmentFeatureId ] == control_treatment.rowId
        )
    # now we remove all invalid cells (according to cellMask) from the mask
    control_treatment_mask = numpy.logical_and( cellMask, control_treatment_mask )

    # extract the features of the control-cells
    controlFeatures = pdc.objFeatures[ control_treatment_mask ]

    # the mahalanobis transformation has difficulties with highly correlated features
    # or features with a low standard-deviation, so we filter those from our list of features
    # (see select_features() for details)
    featureIds, badStddevFeatures, badCorrFeatures = select_features( controlFeatures, featureIds )

    # print out a list of features with bad standard-deviation
    print 'features with bad standard-deviation:'
    print badStddevFeatures

    # print out a list of features which were filtered because of strong correlation
    print 'strongly correlated features:'
    print badCorrFeatures

    # print out how many features are going to be used for differentiation of control-like cells
    print 'using %d features for control cutoff' % len(featureIds)

    # extract the filtered features of the control-cells
    controlFeatures = pdc.objFeatures[ control_treatment_mask ][ : , featureIds ]
    # extract the filtered features of all cells
    testFeatures = pdc.objFeatures[ : , featureIds ]

    # print out a parameter passed to the distance.mahalanobis_distance() function
    print 'mahal_dist_cutoff_fraction=%f' % mahal_dist_cutoff_fraction

    # calculate the mahalanobis distance of all cells to the cells of the control-treatments.
    # thereby only a fraction of the control-cells which are nearest to their own center are used
    # and this fraction is determined by mahal_dist_cutoff_fraction.
    # see distance.mahalanobis_distance() for details.
    mahal_dist = distance.mahalanobis_distance( controlFeatures, testFeatures, mahal_dist_cutoff_fraction )

    # select 'non-normal' cells by ignoring all cells whose mahalanobis distance to the control-group
    # is below a certain threshold.
    #
    # this threshold is determined by the median of the mahalanobis distance of the control cells
    # multiplied by the parameter control_cutoff_threshold
    median_mahal_dist = numpy.median( mahal_dist[ control_treatment_mask ] )
    cutoff_mahal_dist = median_mahal_dist * control_cutoff_threshold
    # the same could also be done with the mean instead of the median
    #mean_mahal_dist = numpy.mean( mahal_dist[ control_treatment_mask ] )
    #cutoff_mahal_dist = mean_mahal_dist * control_cutoff_threshold
    #
    # print out the median (or mean) mahalanobis distance of the control cells
    print 'median=%f' % median_mahal_dist
    #print 'mean=%f' % mean_mahal_dist
    # print out the threshold value for the cutoff
    print 'cutoff_mahal_dist=%f' % cutoff_mahal_dist
    #
    # create a mask according to the cutoff-threshold
    cutoffCellMask = mahal_dist[ : ] > cutoff_mahal_dist

    # extract all the control-like cells (combined with the cellMask of valid cells)
    controlCellMask = numpy.logical_and( cellMask , numpy.logical_not( cutoffCellMask ) )
    # extract all the non-control-like cells (combined with the cellMask of valid cells
    nonControlCellMask = numpy.logical_and( cellMask , cutoffCellMask )

    # print out the number of valid cells
    print 'cells: %d' % numpy.sum( cellMask )
    # print out the number of control-like cells
    print 'control-like cells: %d' % numpy.sum( controlCellMask )
    # print out the number of non-control-like cells
    print 'non-control-like cells: %d' % numpy.sum( nonControlCellMask )

    # extract the features of the non-control-like cells
    objFeatures = pdc.objFeatures[ nonControlCellMask ]
    # now, based only on the non-control-like cells, filter the features again
    featureIds, badStddevFeatures, badCorrFeatures = select_features( objFeatures, featureIds )

    # print out the features with a bad standard-deviation
    print 'features with bad standard-deviation:'
    print badStddevFeatures

    # print out a list of features which were filtered because of strong correlation
    print 'strongly correlated features:'
    print badCorrFeatures

    # print out the number of features to be used for the non-control-like cells
    print 'using %d features' % len(featureIds)

    # return a tuple containing:
    #   - Mask of suspected control cells
    #   - Mask of suspected non-control cells
    #   - List of feature IDs that have been taken into account
    return controlCellMask, nonControlCellMask, featureIds



def select_features( objFeatures, featureIds ):
    """Select features based on their standard-deviation and their correlation coefficients

    Input parameters:
      - objFeatures: Array of feature values. Rows are samples and columns are the features
        corresponding to the feature IDs in 'featureIds'
      - featureIds: List of feature IDs to consider
    Output values (tuple):
      - Array of selected feature IDs
      - List of feature IDs with a 'bad' standard-deviation
      - List of feature IDs with a 'bad' correlation coefficient"""

    newFeatureIds = list( featureIds )


    # don't use features with a low standard-deviation (mahalanobis distance would fail)

    badStddevFeatures = []

    # look at every feature in featureIds...
    for i in xrange( len( newFeatureIds ) ):
        id = newFeatureIds[ i ]
        # calculate the standard-deviation
        stddev = numpy.std( objFeatures[ : , id ] )
        # if the standard-deviation is below the threshold, mark the feature as 'bad'
        if stddev <= min_stddev_threshold:
            badStddevFeatures.append( i )

    # delete all features that were marked as 'bad'
    temp = 0
    for i in badStddevFeatures:
        del newFeatureIds[ i - temp ]
        temp += 1


    # don't use features with a high correlation (mahalanobis distance might fail)

    # calculate the correlation matrix
    ccm = numpy.corrcoef( objFeatures[ : , newFeatureIds ], rowvar=0, bias=0 )

    badCorrFeatures = []

    # look at every pair (i,j) of features in featureIds...
    for i in xrange( ccm.shape[ 0 ] ):
        if i in badCorrFeatures:
            continue
        for j in xrange( i+1, ccm.shape[ 1 ] ):
            if j in badCorrFeatures:
                continue
            # if the correlation coefficient is below the threshold, mark the feature as 'bad'
            if abs( ccm[ i,j ] ) > max_correlation_threshold:
                badCorrFeatures.append( i )
                break

    # delete all features that were marked as 'bad'
    temp = 0
    for i in badCorrFeatures:
        del newFeatureIds[ i - temp ]
        temp += 1

    # return a tuple containing:
    # - array of selected feature IDs
    # - list of feature IDs with a 'bad' standard-deviation
    # - list of feature IDs with a 'bad' correlation coefficient
    return numpy.array( newFeatureIds ), badStddevFeatures, badCorrFeatures

