import sys
import os

import numpy
import scipy
import scipy.linalg

import distance


import parameter_utils as utils

__dict__ = sys.modules[ __name__ ].__dict__

utils.register_module( __name__, 'Analysation of images', __dict__ )

import importer
utils.add_required_state( __name__, importer.__name__, 'imported' )

utils.register_parameter( __name__, 'control_treatment_names', utils.PARAM_TREATMENTS, 'Names of the control treatments' )

#utils.register_parameter( __name__, 'obj_feature_id_names', utils.PARAM_OBJ_FEATURES, 'IDs of the cell features to use' )

utils.register_parameter( __name__, 'mahal_dist_cutoff_fraction', utils.PARAM_FLOAT, 'Fraction of control cells to use for mahalanobis distance', 0.9, 0.0, 1.0 )

utils.register_parameter( __name__, 'min_stddev_threshold', utils.PARAM_FLOAT, 'Minimum threshold for standard deviation of features', 0.01, 0.0, None )

utils.register_parameter( __name__, 'max_correlation_threshold', utils.PARAM_FLOAT, 'Maximum threshold for feature correlation', 0.9, 0.0, 1.0 )

utils.register_parameter( __name__, 'control_cutoff_threshold', utils.PARAM_FLOAT, 'Threshold for cutoff of control-like cells', 1.0, 0.0, None )



def cutoff_control_cells( adc, featureNames, imageMask, cellMask ):

    if len( featureNames ) == 0:
        return None, None, None

    featureIds = []
    for featureName in featureNames:
        featureIds.append( adc.objFeatureIds[ featureName ] )

    imgFeatures = adc.imgFeatures[ imageMask ]
    objFeatures = adc.objFeatures[ cellMask ]


    control_treatment_mask = numpy.empty( (adc.objFeatures.shape[0],) )
    control_treatment_mask[ : ] = False
    for name in control_treatment_names:
        control_treatment = adc.treatments[ adc.treatmentByName[ name ] ]
        control_treatment_mask = numpy.logical_or( control_treatment_mask,
                                                   adc.objFeatures[ : , adc.objTreatmentFeatureId ] == control_treatment.rowId )
    control_treatment_mask = numpy.logical_and( cellMask, control_treatment_mask )

    refMask = control_treatment_mask
    ref = adc.objFeatures[ refMask ]

    featureIds, badStddevFeatures, badCorrFeatures = select_features( ref, featureIds )

    print 'features with bad stddev:'
    print badStddevFeatures

    print 'strongly correlated features:'
    print badCorrFeatures

    print 'using %d features for control cutoff' % len(featureIds)

    ref = adc.objFeatures[ refMask ][ : , featureIds ]
    test = adc.objFeatures[ : , featureIds ]

    print 'mahal_dist_cutoff_fraction=%f' % mahal_dist_cutoff_fraction
    mahal_dist = distance.mahalanobis_distance(ref, test, mahal_dist_cutoff_fraction)

    # select 'non-normal' cells
    median_mahal_dist = numpy.median( mahal_dist[ refMask ] )
    cutoff_mahal_dist = median_mahal_dist * control_cutoff_threshold

    cutoffCellMask = mahal_dist[ : ] > cutoff_mahal_dist

    controlCellMask = numpy.logical_and( cellMask , numpy.logical_not( cutoffCellMask ) )
    nonControlCellMask = numpy.logical_and( cellMask , cutoffCellMask )
    objFeatures = adc.objFeatures[ nonControlCellMask ]

    print 'cells: %d' % numpy.sum( cellMask )
    print 'control-like cells: %d' % numpy.sum( controlCellMask )
    print 'non-control-like cells: %d' % numpy.sum( nonControlCellMask )

    featureIds, badStddevFeatures, badCorrFeatures = select_features( objFeatures, featureIds )

    #objFeatures = adc.objFeatures[ nonControlCellMask ]

    print 'using %d features' % len(featureIds)

    return controlCellMask, nonControlCellMask, featureIds



def select_features( objFeatures, featureIds ):

    newFeatureIds = list( featureIds )


    # don't use features with zero std-deviation (mahalanobis distance would fail)

    badStddevFeatures = []

    for i in xrange( len( newFeatureIds ) ):
        id = newFeatureIds[ i ]
        stddev = numpy.std( objFeatures[ : , id ] )
        if stddev <= min_stddev_threshold:
            badStddevFeatures.append( i )
    temp = 0
    for i in badStddevFeatures:
        del newFeatureIds[ i - temp ]
        temp += 1


    # don't use highly correlated features (mahalanobis distance might fail)

    ccm = numpy.corrcoef( objFeatures[ : , newFeatureIds ], rowvar=0, bias=0 )

    badCorrFeatures = []

    for i in xrange( ccm.shape[ 0 ] ):
        if i in badCorrFeatures:
            continue
        for j in xrange( i+1, ccm.shape[ 1 ] ):
            if j in badCorrFeatures:
                continue
            if abs( ccm[ i,j ] ) > max_correlation_threshold:
                badCorrFeatures.append( i )
                break
    temp = 0
    for i in badCorrFeatures:
        del newFeatureIds[ i - temp ]
        temp += 1

    return numpy.array( newFeatureIds ), badStddevFeatures, badCorrFeatures

