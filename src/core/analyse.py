# -*- coding: utf-8 -*-

"""
analyse.py -- Preliminary analysis of the cells.

Feature selection and the cutoff of control cells is done here. See
    reject_cells_with_bad_features()
    reject_control_cells()
"""

# This software is distributed under the FreeBSD License.
# See the accompanying file LICENSE for details.
# 
# Copyright 2011 Benjamin Hepp

import sys
import os
import itertools

import numpy
import scipy
import scipy.linalg
import sklearn.decomposition

import distance


# define necessary parameters for this module (see parameter_utils.py for details)
#
import parameter_utils as utils
#
__dict__ = sys.modules[__name__].__dict__
#
utils.register_module(__name__, 'Analysis of images', __dict__)
#
import importer
utils.add_required_state(__name__, importer.__name__, 'imported')
#
utils.register_parameter(__name__, 'control_treatment_names', utils.PARAM_TREATMENTS, 'Names of the control treatments')
#
utils.register_parameter(__name__, 'run_mahal_dist', utils.PARAM_BOOL, 'Run Mahalanobis filter?', False)
#
utils.register_parameter(__name__, 'mahal_dist_cutoff_fraction', utils.PARAM_FLOAT, 'Fraction of control cells to use for mahalanobis distance', 0.9, 0.0, 1.0)
#
utils.register_parameter(__name__, 'threshold_fraction', utils.PARAM_FLOAT, 'Fraction considered as showing a phenotype', 0.2, 0.0, 1.0)
#
utils.register_parameter(__name__, 'min_stddev_threshold', utils.PARAM_FLOAT, 'Minimum threshold for standard deviation of features', 0.01, 0.0, None)
#
utils.register_parameter(__name__, 'max_correlation_threshold', utils.PARAM_FLOAT, 'Maximum threshold for feature correlation', 0.9, 0.0, 1.0)
#
utils.register_parameter(__name__, 'control_cutoff_threshold', utils.PARAM_FLOAT, 'Threshold for cutoff of control-like cells', 1.0, -1.0, None)
#
utils.register_parameter(__name__, 'filter_feature_set', utils.PARAM_INT, 'Feature set to use for pre-filtering', -1, -1, 100)
#
utils.register_parameter(__name__, 'filter_obj_feature_id_names', utils.PARAM_OBJ_FEATURES, 'Features to use for pre-filtering', [])
#
utils.register_parameter(__name__, 'mean_intensity_feature_id_names', utils.PARAM_OBJ_FEATURES, 'Feature IDs for the mean intensities of a cell', [])
#
#utils.register_parameter(__name__, 'intensity_normalize_feature_id_names', utils.PARAM_OBJ_FEATURES, 'Features to normalize according to the mean intensity of the control cells', [])


def normalize_to_control_cell_intensity(pdc, validCellMask):

    # create a mask that contains all the cells in every control treatment
    #
    # we start with an empty mask
    control_treatment_mask = numpy.empty((pdc.objFeatures.shape[0],))
    control_treatment_mask[:] = False
    # and for each control treatment...
    for name in control_treatment_names:
        control_treatment = pdc.treatments[pdc.treatmentByName[name]]
        # we add the corresponding cells to the mask
        control_treatment_mask = numpy.logical_or(
                control_treatment_mask,
                pdc.objFeatures[: , pdc.objTreatmentFeatureId] == control_treatment.index
       )
    # now we remove all invalid cells (according to cellMask) from the mask
    control_treatment_mask = numpy.logical_and(validCellMask, control_treatment_mask)

    #control_mean_intensity = numpy.mean(pdc.objFeatures[control_treatment_mask][: , mean_intensity_feature_id])
    print 'mean intensity of control cells:', control_mean_intensity

    # determine the feature IDs corresponding to the feature names in filter_obj_feature_id_names
    featureIds = []
    for featureName in intensity_normalize_feature_id_names:
        featureIds.append(pdc.objFeatureIds[featureName])

    for featureId in featureIds:
        objFeatures[: , featureId] = objFeatures[: , featureId] / control_mean_intensity

def output_mean_control_cell_intensities(pdc, validCellMask):

    # create a mask that contains all the cells in every control treatment
    #
    # we start with an empty mask
    control_treatment_mask = numpy.empty((pdc.objFeatures.shape[0],))
    control_treatment_mask[:] = False
    # and for each control treatment...
    for name in control_treatment_names:
        control_treatment = pdc.treatments[pdc.treatmentByName[name]]
        # we add the corresponding cells to the mask
        control_treatment_mask = numpy.logical_or(
                control_treatment_mask,
                pdc.objFeatures[: , pdc.objTreatmentFeatureId] == control_treatment.index
       )
    # now we remove all invalid cells (according to cellMask) from the mask
    control_treatment_mask = numpy.logical_and(validCellMask, control_treatment_mask)

    # determine the feature IDs corresponding to the feature names in mean_intensity_feature_id_names
    featureIdNameTuples = []
    for featureName in mean_intensity_feature_id_names:
        featureIdNameTuples.append((pdc.objFeatureIds[featureName], featureName))

    meanIntensities = []
    for featureId,featureName in featureIdNameTuples:
        meanIntensity = numpy.mean(pdc.objFeatures[control_treatment_mask][: , featureId])
        print 'mean intensity of %s (%d): %f' % (featureName, featureId, meanIntensity)

def run_pca(pdc, cellMask, featureIds, n_components, kernel='rbf'):
    data = pdc.objFeatures[cellMask][:, featureIds]
    #pca = sklearn.decomposition.PCA(n_components)
    #pca = sklearn.decomposition.KernelPCA(n_components, kernel=kernel)
    pca = sklearn.decomposition.FastICA(n_components)
    pca.fit(data)
    return pca

def reject_cells_with_bad_features(pdc, imageMask, cellMask, featureNames=None):
    """Reject cells which have NaN or Inf values for features that we want to use

    Input parameters:
      - pdc: YACA data container
      - imageMask: Mask of images to take into account
      - cellMask: Mask of cells to take into account
    Output values (tuple):
      - Mask of cells with no bad feature values
      - A list of feature IDs that have been taken into account
      - A numpy array specifying for each feature the number of
        cells which had a bad value
    """

    if featureNames == None:
        featureNames = filter_obj_feature_id_names

    # trivial check
    # if filter_obj_feature_id_names is empty, there are no bad feature values
    if len(featureNames) == 0:
        return cellMask, [], numpy.zeros((0,))

    # determine the feature IDs corresponding to the feature names in filter_obj_feature_id_names
    featureIds = []
    for featureName in featureNames:
        featureIds.append(pdc.objFeatureIds[featureName])

    nan_mask = numpy.isnan(pdc.objFeatures[:, featureIds])
    inverse_cell_mask = numpy.invert(cellMask)
    nan_mask[inverse_cell_mask, :] = False
    bad_feature_cell_count = numpy.sum(nan_mask, axis=0)

    collapsed_nan_mask = numpy.any(nan_mask, axis=1)
    inverse_mask = numpy.invert(collapsed_nan_mask)

    cellMask = numpy.logical_and(inverse_mask, cellMask)

    return cellMask, featureIds, bad_feature_cell_count


def reject_control_treatments(pdc, cellMask, nonControlCellMask):
    """Make a heuristic guess about which treatments are control-like and which treatments show a phenotype

    Input parameters:
      - pdc: YACA data container
      - cellMask: Mask of valid cell objects
      - nonControlCellMask: Mask of non-control-like cell objects

    Output values (tuple):
      - List of treatments showing a phenotype
    """

    fractions = []
    treatmentId = pdc.objFeatures[:, pdc.objTreatmentFeatureId]
    for tr in pdc.treatments:
        tr_mask = treatmentId == tr.index
        fraction = numpy.sum(numpy.logical_and(nonControlCellMask, tr_mask)) / \
            float(numpy.sum(numpy.logical_and(cellMask, tr_mask)))
        fractions.append(fraction)
        print 'tr: %s -> %f' % (tr.name, fraction)
    return itertools.ifilter(lambda tr: fractions[tr.index] > threshold_fraction, pdc.treatments)

FILTER_MODE_MEDIAN_xMAD             = 0
FILTER_MODE_xMEDIAN                 = 1
FILTER_MODE_MEDIAN_xMAD_AND_LIMIT   = 2
FILTER_MODE_xMEDIAN_AND_LIMIT       = 3

def reject_control_cells(pdc, imageMask, cellMask, control_treatment_mask, featureIds, filterMode=FILTER_MODE_MEDIAN_xMAD, pca=None):
    """Make a heuristic guess about which cells are control-like cells and which cells are not

    Input parameters:
      - pdc: YACA data container
      - imageMask: Mask of images to take into account
      - cellMask: Mask of cells to take into account
    Output values (tuple):
      - Mask of suspected control cells
      - Mask of suspected non-control cells
      - List of feature IDs that have been taken into account
    """

    """# trivial check
    # if filter_obj_feature_id_names is empty, assume that all cells are non-control like cells
    if featureNames == None:
        featureNames = filter_obj_feature_id_names
    if len(featureNames) == 0:
        emptyMask = numpy.empty((pdc.objFeatures.shape[0],))
        emptyMask[:] = False
        mahal_dist = numpy.empty((pdc.objFeatures.shape[0],))
        mahal_dist[:] = numpy.nan
        return emptyMask, cellMask, [], mahal_dist, {}

    # determine the feature IDs corresponding to the feature names in filter_obj_feature_id_names
    featureIds = []
    for featureName in featureNames:
        featureId = pdc.objFeatureIds[featureName]
        #print 'Using feature %d -> %s' % (featureId, featureName)
        featureIds.append(featureId)"""

    if len(featureIds) == 0:
        emptyMask = numpy.empty((pdc.objFeatures.shape[0],))
        emptyMask[:] = False
        mahal_dist = numpy.empty((pdc.objFeatures.shape[0],))
        mahal_dist[:] = numpy.nan
        return emptyMask, cellMask, [], mahal_dist, {}


    # print out some info
    print 'number of selected features for phenotypic filtering: %d' % len(featureIds)

    # extract the image- and object-features
    #imgFeatures = pdc.imgFeatures[imageMask]
    #objFeatures = pdc.objFeatures[cellMask]

    """# extract the features of the control-cells
    #controlFeatures = pdc.objFeatures[control_treatment_mask]

    # the mahalanobis transformation has difficulties with highly correlated features
    # or features with a low standard-deviation, so we filter those from our list of features
    # (see select_features() for details)
    featureIds, badNaNFeatures, badStddevFeatures, badCorrFeatures = select_features(objFeatures], featureIds)

    # print out the features with NaN values
    #print 'features with NaN values:'
    #for fid in badNaNFeatures:
        #print '%d -> %s' % (fid, pdc.objFeatureName(fid))
    print 'number of features with NaN values: %d' % len(badNaNFeatures)

    # print out a list of features with bad standard-deviation
    #print 'features with bad standard-deviation:'
    #for fid in badStddevFeatures:
        #print '%d -> %s' % (fid, pdc.objFeatureName(fid))
    print 'number of features with bad standard-deviation: %d' % len(badStddevFeatures)

    ## print out a list of features which were filtered because of strong correlation
    #print 'strongly correlated features:'
    #for fid in badCorrFeatures:
        #print '%d -> %s' % (fid, pdc.objFeatureName(fid))
    print 'number of features with strong correlation: %d' % len(badCorrFeatures)

    # print out how many features are going to be used for differentiation of control-like cells
    print 'using %d features for control cutoff' % len(featureIds)"""

    #for featureId in featureIds:
        #featureName = pdc.objFeatureName(featureId)
        #print ' %d -> %s' % (featureId, featureName)

    # extract the filtered features of the control-cells
    controlFeatures = pdc.objFeatures[numpy.logical_and(cellMask, control_treatment_mask)][: , featureIds]
    if pca is not None:
        controlFeatures = pca.transform(controlFeatures)
    # extract the filtered features of all cells
    testFeatures = pdc.objFeatures[cellMask][:, featureIds]
    if pca is not None:
        testFeatures = pca.transform(testFeatures)

    # print out a parameter passed to the distance.mahalanobis_distance() function
    print 'mahal_dist_cutoff_fraction=%f' % mahal_dist_cutoff_fraction

    print 'controlFeatures.shape:', controlFeatures.shape
    print 'testFeatures.shape:', testFeatures.shape

    # calculate the mahalanobis distance of all cells to the cells of the control-treatments.
    # thereby only a fraction of the control-cells which are nearest to their own center are used
    # and this fraction is determined by mahal_dist_cutoff_fraction.
    # see distance.mahalanobis_distance() for details.
    mahal_dist = distance.mahalanobis_distance(controlFeatures, testFeatures, mahal_dist_cutoff_fraction)
    mahal_dist = numpy.sqrt(mahal_dist)
    new_mahal_dist = numpy.empty((pdc.objFeatures.shape[0],))
    new_mahal_dist[cellMask] = mahal_dist
    new_mahal_dist[numpy.invert(cellMask)] = numpy.inf
    mahal_dist = new_mahal_dist


    # select 'non-normal' cells by ignoring all cells whose mahalanobis distance to the control-group
    # is below a certain threshold.
    #
    # this threshold is determined by the median of the mahalanobis distance of the control cells
    # multiplied by the parameter control_cutoff_threshold
    ctrl_median_mahal_dist = numpy.median(mahal_dist[control_treatment_mask])
    ctrl_mad_mahal_dist = numpy.median(numpy.abs(mahal_dist[control_treatment_mask] - ctrl_median_mahal_dist))
    # compute upper_cutoff_threshold as median + <control_cutoff_threshold> * mad
    #     and lower_cutoff_threshold as median - <control_cutoff_threshold> * mad
    if filterMode == FILTER_MODE_MEDIAN_xMAD or filterMode == FILTER_MODE_MEDIAN_xMAD_AND_LIMIT:
        upper_cutoff_mahal_dist = ctrl_median_mahal_dist + control_cutoff_threshold * ctrl_mad_mahal_dist
        lower_cutoff_mahal_dist = ctrl_median_mahal_dist - control_cutoff_threshold * ctrl_mad_mahal_dist
        lower_cutoff_mahal_dist = 0.0
    elif filterMode == FILTER_MODE_xMEDIAN or filterMode == FILTER_MODE_xMEDIAN_AND_LIMIT:
        upper_cutoff_mahal_dist = + control_cutoff_threshold * ctrl_median_mahal_dist
        lower_cutoff_mahal_dist = 0.0
    if not run_mahal_dist:
        upper_cutoff_mahal_dist = 0.0
        lower_cutoff_mahal_dist = 0.0
    # the same could also be done with the mean instead of the median
    #mean_mahal_dist = numpy.mean(mahal_dist[control_treatment_mask])
    #cutoff_mahal_dist = mean_mahal_dist * control_cutoff_threshold
    #
    # print out the median (or mean) mahalanobis distance of the control cells
    print 'median_mahal_dist=%f' % ctrl_median_mahal_dist
    print 'mad_mahal_dist=%f' % ctrl_mad_mahal_dist
    #print 'mean=%f' % mean_mahal_dist
    # print out the threshold value for the cutoff
    print 'upper_cutoff_mahal_dist=%f' % upper_cutoff_mahal_dist
    print 'lower_cutoff_mahal_dist=%f' % lower_cutoff_mahal_dist

    #import math
    #print 'sqrt(median_mahal_dist)=%f' % math.sqrt(ctrl_median_mahal_dist)
    #print 'sqrt(mad_mahal_dist)=%f' % math.sqrt(ctrl_mad_mahal_dist)
    #print 'sqrt(cutoff_mahal_dist)=%f' % math.sqrt(cutoff_mahal_dist)

    # create a mask according to the cutoff-threshold
    cutoffCellMask = numpy.logical_or(mahal_dist[:] > upper_cutoff_mahal_dist,
                                      mahal_dist[:] < lower_cutoff_mahal_dist)
    # extract all the control-like cells (combined with the cellMask of valid cells)
    controlCellMask = numpy.logical_and(cellMask , numpy.logical_not(cutoffCellMask))
    # extract all the non-control-like cells (combined with the cellMask of valid cells
    nonControlCellMask = numpy.logical_and(cellMask , cutoffCellMask)

    # if more than 10% of the control cells are above the cutoff_mahal_dist, increase the cutoff_mahal_dist so that
    # less than 10% of the control cells are above the cutoff_mahal_dist
    """print numpy.sum(nonControlCellMask) / float(numpy.sum(cellMask))
    for ctrl_tr_name in control_treatment_names:
        ctrl_tr = pdc.treatments[pdc.treatmentByName[ctrl_tr_name]]
        ctrl_tr_mask =  numpy.logical_and(cellMask, pdc.objFeatures[:, pdc.objTreatmentFeatureId] == ctrl_tr.index)
        mask1 = numpy.logical_and(nonControlCellMask, ctrl_tr_mask)
        print ctrl_tr_name
        print (numpy.sum(mask1) / float(numpy.sum(ctrl_tr_mask)))
        if numpy.sum(mask1) / float(numpy.sum(ctrl_tr_mask)) > 0.1:
            values = numpy.sort(mahal_dist[ctrl_tr_mask])
            index = int(0.9 * values.shape[0] + 1)
            index = min(index, values.shape[0] - 1)
            print cutoff_mahal_dist
            cutoff_mahal_dist = values[index]
            print cutoff_mahal_dist

        # create a mask according to the cutoff-threshold
        cutoffCellMask = mahal_dist[:] > cutoff_mahal_dist
        # extract all the control-like cells (combined with the cellMask of valid cells)
        controlCellMask = numpy.logical_and(cellMask , numpy.logical_not(cutoffCellMask))
        # extract all the non-control-like cells (combined with the cellMask of valid cells
        nonControlCellMask = numpy.logical_and(cellMask , cutoffCellMask)"""

    if control_cutoff_threshold >= 0.0 and \
       (filterMode == FILTER_MODE_xMEDIAN_AND_LIMIT or \
        filterMode == FILTER_MODE_MEDIAN_xMAD_AND_LIMIT):

        mask1 = numpy.logical_and(nonControlCellMask, control_treatment_mask)
        print (numpy.sum(mask1) / float(numpy.sum(control_treatment_mask)))
        if numpy.sum(mask1) / float(numpy.sum(control_treatment_mask)) > 0.1:
            values = numpy.sort(mahal_dist[control_treatment_mask])
            index = int(0.9 * values.shape[0] + 1)
            index = min(index, values.shape[0] - 1)
            upper_cutoff_mahal_dist = values[index]
            lower_cutoff_mahal_dist = 0.0

    print 'upper_cutoff_mahal_dist=%f' % upper_cutoff_mahal_dist
    print 'lower_cutoff_mahal_dist=%f' % lower_cutoff_mahal_dist

    # create a mask according to the cutoff-threshold
    cutoffCellMask = numpy.logical_or(mahal_dist[:] > upper_cutoff_mahal_dist,
                                      mahal_dist[:] < lower_cutoff_mahal_dist)
    # extract all the control-like cells (combined with the cellMask of valid cells)
    controlCellMask = numpy.logical_and(cellMask , numpy.logical_not(cutoffCellMask))
    # extract all the non-control-like cells (combined with the cellMask of valid cells
    nonControlCellMask = numpy.logical_and(cellMask , cutoffCellMask)


    """treatment_mask_1 = numpy.ones((len(pdc.treatments),), dtype=bool)
    print 'mahalanobis medians and mads of treatments..'
    for i in xrange(len(pdc.treatments)):
        tr = pdc.treatments[i]
        tr_mask = numpy.logical_and(cellMask, pdc.objFeatures[:, pdc.objTreatmentFeatureId] == tr.index)
        median_mahal_dist = numpy.median(mahal_dist[tr_mask])
        mad_mahal_dist = numpy.median(numpy.abs(mahal_dist[tr_mask] - median_mahal_dist))
        if abs(median_mahal_dist - ctrl_median_mahal_dist) < (mad_mahal_dist + ctrl_mad_mahal_dist):
            print 'treatment %s might not show a phenotype!!!' % tr.name
            treatment_mask_1[i] = False
        print '  %s: %f +- %f' % (tr.name, median_mahal_dist, mad_mahal_dist)
        cell_selection_stats['median_mahal_dist'].append(median_mahal_dist)
        cell_selection_stats['mad_mahal_dist'].append(mad_mahal_dist)

    treatment_mask_2 = numpy.ones((len(pdc.treatments),), dtype=bool)
    print 'mahalanobis medians and mads of cutoff treatments..'
    for i in xrange(len(pdc.treatments)):
        tr = pdc.treatments[i]
        tr_mask = numpy.logical_and(nonControlCellMask, pdc.objFeatures[:, pdc.objTreatmentFeatureId] == tr.index)
        median_mahal_dist = numpy.median(mahal_dist[tr_mask])
        mad_mahal_dist = numpy.median(numpy.abs(mahal_dist[tr_mask] - median_mahal_dist))
        if abs(median_mahal_dist - ctrl_median_mahal_dist) < (mad_mahal_dist + ctrl_mad_mahal_dist):
            print 'treatment %s might not show a phenotype!!!' % tr.name
            treatment_mask_2[i] = False
        print '  %s: %f +- %f' % (tr.name, median_mahal_dist, mad_mahal_dist)
        cell_selection_stats['median_mahal_cutoff_dist'].append(median_mahal_dist)
        cell_selection_stats['mad_mahal_cutoff_dist'].append(mad_mahal_dist)

    treatment_mask_3 = numpy.ones((len(pdc.treatments),), dtype=bool)
    print 'penetrance of treatments..'
    for i in xrange(len(pdc.treatments)):
        tr = pdc.treatments[i]
        tr_mask = numpy.logical_and(cellMask, pdc.objFeatures[:, pdc.objTreatmentFeatureId] == tr.index)
        nonCtrl_tr_mask = numpy.logical_and(nonControlCellMask, pdc.objFeatures[:, pdc.objTreatmentFeatureId] == tr.index)
        penetrance = numpy.sum(nonCtrl_tr_mask / float(numpy.sum(tr_mask)))
        if penetrance < 0.1:
            print 'treatment %s might not show a phenotype!!!' % tr.name
            treatment_mask_3[i] = False
        print '  %s: %f %%' % (tr.name, penetrance * 100)

    treatment_mask = numpy.logical_and(treatment_mask_2, numpy.logical_or(treatment_mask_1, treatment_mask_3))"""

    #print 'phenotypes...'
    #for i in xrange(len(pdc.treatments)):
    #    tr = pdc.treatments[i]
    #    if treatment_mask[i]:
    #        print '  %s shows a phenotype' % tr.name

    # print out the number of valid cells
    print 'cells: %d' % numpy.sum(cellMask)
    # print out the number of control-like cells
    print 'control-like cells: %d' % numpy.sum(controlCellMask)
    # print out the number of non-control-like cells
    print 'non-control-like cells: %d' % numpy.sum(nonControlCellMask)

    """# extract the features of the non-control-like cells
    objFeatures = pdc.objFeatures[nonControlCellMask]
    # now, based only on the non-control-like cells, filter the features again
    featureIds, badNaNFeatures, badStddevFeatures, badCorrFeatures = select_features(objFeatures, featureIds)

    # print out the features with NaN values
    print 'features with NaN values:'
    print badNaNFeatures

    # print out the features with a bad standard-deviation
    print 'features with bad standard-deviation:'
    print badStddevFeatures

    # print out a list of features which were filtered because of strong correlation
    print 'strongly correlated features:'
    print badCorrFeatures

    # print out the number of features to be used for the non-control-like cells
    print 'using %d features' % len(featureIds)"""

    # return a tuple containing:
    #   - Mask of suspected control cells
    #   - Mask of suspected non-control cells
    #   - List of feature IDs that have been taken into account
    return controlCellMask, nonControlCellMask, featureIds, mahal_dist, (lower_cutoff_mahal_dist, upper_cutoff_mahal_dist)



def select_features(objFeatures, featureIds):
    """Select features based on their standard-deviation and their correlation coefficients

    Input parameters:
      - objFeatures: Array of feature values. Rows are samples and columns are the features
        corresponding to the feature IDs in 'featureIds'
      - featureIds: List of feature IDs to consider
    Output values (tuple):
      - Array of selected feature IDs
      - List of feature IDs with a 'bad' standard-deviation
      - List of feature IDs with a 'bad' correlation coefficient"""

    newFeatureIds = list(featureIds)


    # don't use features with a NaN-values

    badNaNFeatures = []
    badNaNFeatureIds = []

    # look at every feature in featureIds...
    for i in xrange(len(newFeatureIds)):
        fid = newFeatureIds[i]
        # mask NaN values
        nan_mask = numpy.isnan(objFeatures[: , fid])
        # if any value is NaN, mark the feature as 'bad'
        if numpy.any(nan_mask):
            badNaNFeatures.append(i)
            badNaNFeatureIds.append(fid)

    # delete all features that were marked as 'bad'
    temp = 0
    for i in badNaNFeatures:
        del newFeatureIds[i - temp]
        temp += 1


    # don't use features with a low standard-deviation (mahalanobis distance would fail)

    badStddevFeatures = []
    badStddevFeatureIds = []

    # look at every feature in featureIds...
    for i in xrange(len(newFeatureIds)):
        fid = newFeatureIds[i]
        # calculate the standard-deviation
        stddev = numpy.std(objFeatures[: , fid])
        # if the standard-deviation is below the threshold, mark the feature as 'bad'
        if stddev <= min_stddev_threshold:
            badStddevFeatures.append(i)
            badStddevFeatureIds.append(fid)

    # delete all features that were marked as 'bad'
    temp = 0
    for i in badStddevFeatures:
        del newFeatureIds[i - temp]
        temp += 1


    # don't use features with a high correlation (mahalanobis distance might fail)

    # calculate the correlation matrix
    ccm = numpy.corrcoef(objFeatures[: , newFeatureIds], rowvar=0, bias=0)
    if type(ccm) == int:
        ccm = numpy.array([ccm])

    badCorrFeatures = []
    badCorrFeatureIds = []

    # look at every pair (i,j) of features in featureIds...
    for (i, fid1) in enumerate(newFeatureIds):
    #for i in xrange(ccm.shape[0]):
        if i in badCorrFeatures:
            continue
        for (j, fid2) in enumerate(newFeatureIds[i+1 :]):
        #for j in xrange(i+1, ccm.shape[0]):
            if j in badCorrFeatures:
                continue
            # if the correlation coefficient is below the threshold, mark the feature as 'bad'
            if abs(ccm[i,j]) > max_correlation_threshold:
                badCorrFeatures.append(i)
                badCorrFeatureIds.append(fid1)
                break

    # delete all features that were marked as 'bad'
    temp = 0
    for i in badCorrFeatures:
        del newFeatureIds[i - temp]
        temp += 1

    # return a tuple containing:
    # - array of selected feature IDs
    # - list of feature IDs with NaN values
    # - list of feature IDs with a 'bad' standard-deviation
    # - list of feature IDs with a 'bad' correlation coefficient
    return numpy.array(newFeatureIds), badNaNFeatureIds, badStddevFeatureIds, badCorrFeatureIds
