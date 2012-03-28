# -*- coding: utf-8 -*-

"""
quality_control.py -- Quality control of images and cells.

Images and cells have to fullfil certain properties to be seen as valid cells.

- quality_control() performs these checks and marks images and cells as valid or invalid.

See the documentation for further details.
"""

# This software is distributed under the FreeBSD License.
# See the accompanying file LICENSE for details.
# 
# Copyright 2011 Benjamin Hepp

import sys
import numpy

import quality_control_mp

QUALITY_CONTROL_DEFAULT = 0
QUALITY_CONTROL_VALID = 1

QUALITY_CONTROL_NOT_ENOUGH_VALID_CELLS = 1001
QUALITY_CONTROL_TOO_MANY_CELLS = 1002
QUALITY_CONTROL_NOT_ENOUGH_BG_PIXELS = 1003

QUALITY_CONTROL_OUT_OF_PERIPHERY = 2001
QUALITY_CONTROL_TOO_SMALL_NUCLEUS = 2002
QUALITY_CONTROL_TOO_BIG_NUCLEUS = 2003
QUALITY_CONTROL_TOO_SMALL_CYTOPLASM = 2004
QUALITY_CONTROL_TOO_BIG_CYTOPLASM = 2005
QUALITY_CONTROL_TOO_SMALL_NUCLEUS_SOLIDITY = 2006

QUALITY_CONTROL_DESCR = {
    QUALITY_CONTROL_DEFAULT : 'default',
    QUALITY_CONTROL_VALID : 'valid',
    QUALITY_CONTROL_OUT_OF_PERIPHERY : 'out of image periphery',
    QUALITY_CONTROL_TOO_SMALL_NUCLEUS : 'too small nucleus area',
    QUALITY_CONTROL_TOO_BIG_NUCLEUS : 'too big nucleus area',
    QUALITY_CONTROL_TOO_SMALL_CYTOPLASM : 'too small cytoplasm area',
    QUALITY_CONTROL_TOO_BIG_CYTOPLASM : 'too big cytoplasm area',
    QUALITY_CONTROL_TOO_SMALL_NUCLEUS_SOLIDITY : 'too small nucleus solidity',
    QUALITY_CONTROL_NOT_ENOUGH_VALID_CELLS : 'not enough valid cells',
    QUALITY_CONTROL_TOO_MANY_CELLS : 'too many cells',
    QUALITY_CONTROL_NOT_ENOUGH_BG_PIXELS : 'not enough background pixels'
}

import parameter_utils as utils

__dict__ = sys.modules[__name__].__dict__

utils.register_module(__name__, 'Quality control of cells and images', __dict__)


import importer
utils.add_required_state(__name__, importer.__name__, 'imported')

utils.register_parameter(__name__, 'positionX', utils.PARAM_OBJ_FEATURE, 'Feature ID for the x coordinate of a cell (in pixel)', 'nucleus_Location_Center_X')
utils.register_parameter(__name__, 'positionY', utils.PARAM_OBJ_FEATURE, 'Feature ID for the y coordinate of a cell (in pixel)', 'nucleus_Location_Center_Y')

utils.register_parameter(__name__, 'cellArea', utils.PARAM_OBJ_FEATURE, 'Feature ID for the area of a cell (in pixel)', 'cell_AreaShape_Area')
utils.register_parameter(__name__, 'nucleusArea', utils.PARAM_OBJ_FEATURE, 'Feature ID for the area of a nucleus (in pixel)', 'nucleus_AreaShape_Area')

utils.register_parameter(__name__, 'nucleusSolidity', utils.PARAM_OBJ_FEATURE, 'Feature ID for the solidity of a nucleus', 'nucleus_AreaShape_Solidity')

utils.register_parameter(__name__, 'minAreaBg', utils.PARAM_INT, 'Minimum number of background pixels', 50000, 0, None)
utils.register_parameter(__name__, 'minNucArea', utils.PARAM_INT, 'Minimum area of nuclei in pixels', 1000, 0, None)
utils.register_parameter(__name__, 'maxNucArea', utils.PARAM_INT, 'Maximum area of nuclei in pixels', 4000, 0, None)

utils.register_parameter(__name__, 'minNucSolidity', utils.PARAM_FLOAT, 'Minimum nuclei solidity', 0.9, 0.0, 1.0)

utils.register_parameter(__name__, 'minDistLeft', utils.PARAM_INT, 'Minimum distance of cells to left image margin', 100, 0, None)
utils.register_parameter(__name__, 'minDistRight', utils.PARAM_INT, 'Minimum distance of cells to right image margin', 100, 0, None)
utils.register_parameter(__name__, 'minDistTop', utils.PARAM_INT, 'Minimum distance of cells top image margin', 100, 0, None)
utils.register_parameter(__name__, 'minDistBottom', utils.PARAM_INT, 'Minimum distance of cells to bottom image margin', 100, 0, None)

utils.register_parameter(__name__, 'imageWidth', utils.PARAM_INT, 'Width of images in pixel', 1344, 1, None)
utils.register_parameter(__name__, 'imageHeight', utils.PARAM_INT, 'Height of images in pixel', 1024, 1, None)

utils.register_parameter(__name__, 'minCyToNucAreaFrac', utils.PARAM_FLOAT, 'Minimum allowed fraction of cytoplasma to nucleus area', 1.5, 0.0, None)
utils.register_parameter(__name__, 'maxCyToNucAreaFrac', utils.PARAM_FLOAT, 'Maximum allowed fraction of cytoplasma to nucleus area', 15, 0.0, None)

utils.register_parameter(__name__, 'minCells', utils.PARAM_INT, 'Minimum number of cells per image', 1, 1, None)
utils.register_parameter(__name__, 'maxCells', utils.PARAM_INT, 'Maximum number of cells per image', 300, 1, None)

utils.register_parameter(__name__, 'numOfProcesses', utils.PARAM_INT, 'Number of parallel processes', 10, 1, None)


# quality control of the data
def quality_control(pdc):

    print 'cellArea: %s' % cellArea
    print 'nucleusArea: ', nucleusArea
    print 'positionX: ', positionX
    print 'positionY: ', positionY

    validImageMask = numpy.empty((len(pdc.images),) , dtype=bool)
    validImageMask[:] = True

    validCellMask = numpy.empty((len(pdc.objects),) , dtype=bool)
    validCellMask[:] = True

    qualityControlFeature = numpy.empty((len(pdc.objects),))

    if pdc.objQualityControlFeatureId >= 0:
        pdc.objFeatures[: , pdc.objQualityControlFeatureId] = QUALITY_CONTROL_DEFAULT
    if pdc.imgQualityControlFeatureId >= 0:
        pdc.imgFeatures[: , pdc.imgQualityControlFeatureId] = QUALITY_CONTROL_DEFAULT


    # cells are at image periphery
    #print "\n".join(pdc.objFeatureIds.keys())
    featureIdX = pdc.objFeatureIds[positionX]
    featureIdY = pdc.objFeatureIds[positionY]
    n1 = validCellMask.sum()
    mask = pdc.objFeatures[: , featureIdX] > minDistLeft
    mask = numpy.logical_and(mask,
                (pdc.objFeatures[: , featureIdX] < imageWidth-minDistRight)
   )
    mask = numpy.logical_and(mask,
                (pdc.objFeatures[: , featureIdY] > minDistTop)
   )
    mask = numpy.logical_and(mask,
                (pdc.objFeatures[: , featureIdY] < imageHeight-minDistBottom)
   )
    if pdc.objQualityControlFeatureId >= 0:
        failMask = numpy.invert(mask)
        qualityControlFeature[numpy.logical_and(validCellMask, failMask)] = QUALITY_CONTROL_OUT_OF_PERIPHERY
    validCellMask = numpy.logical_and(validCellMask, mask)
    n2 = validCellMask.sum()
    print 'found %d cells out of image periphery' % (n1 - n2)

    # minimum nucleus area
    nucleusFeatureId = pdc.objFeatureIds[nucleusArea]
    #thresholdNucleusArea = numpy.median(image.objFeatures[:,featureId] \
    #              - 2 * numpy.mean(numpy.abs(
    #                      image.objFeatures[:,featureId]
    #                      - numpy.mean(image.objFeatures[:,featureId])
    #             ))
    #)
    mask = pdc.objFeatures[: , nucleusFeatureId] > minNucArea
    if pdc.objQualityControlFeatureId >= 0:
        failMask = numpy.invert(mask)
        qualityControlFeature[numpy.logical_and(validCellMask, failMask)] = QUALITY_CONTROL_TOO_SMALL_NUCLEUS
    validCellMask = numpy.logical_and(validCellMask, mask)
    n3 = validCellMask.sum()
    print 'found %d cells with minimum nucleus area' % (n2 - n3)

    # maximum nucleus area
    mask = pdc.objFeatures[: , nucleusFeatureId] < maxNucArea
    if pdc.objQualityControlFeatureId >= 0:
        failMask = numpy.invert(mask)
        qualityControlFeature[numpy.logical_and(validCellMask, failMask)] = QUALITY_CONTROL_TOO_BIG_NUCLEUS
    validCellMask = numpy.logical_and(validCellMask, mask)
    n4 = validCellMask.sum()
    print 'found %d cells with maximum nucleus area' % (n3 - n4)

    # minimum relative cytoplasm area
    #cellFeatureId = pdc.cellFeatureIds['AreaShape_Area']
    cellFeatureId = pdc.objFeatureIds[cellArea]
    mask = pdc.objFeatures[: , cellFeatureId] \
                          > (minCyToNucAreaFrac * pdc.objFeatures[: , nucleusFeatureId] )
    if pdc.objQualityControlFeatureId >= 0:
        failMask = numpy.invert(mask)
        qualityControlFeature[numpy.logical_and(validCellMask, failMask)] = QUALITY_CONTROL_TOO_SMALL_CYTOPLASM
    validCellMask = numpy.logical_and(validCellMask, mask)
    n5 = validCellMask.sum()
    print 'found %d cells with minimum relative cytoplasm area' % (n4 - n5)

    # maximum relative cytoplasm area
    mask = (pdc.objFeatures[: , cellFeatureId] \
                  < maxCyToNucAreaFrac * pdc.objFeatures[: , nucleusFeatureId])
    if pdc.objQualityControlFeatureId >= 0:
        failMask = numpy.invert(mask)
        qualityControlFeature[numpy.logical_and(validCellMask, failMask)] = QUALITY_CONTROL_TOO_BIG_CYTOPLASM
    validCellMask = numpy.logical_and(validCellMask, mask)
    n6 = validCellMask.sum()
    print 'found %d cells with maximum relative cytoplasm area' % (n5 - n6)

    # minimum nucleus solidity
    nucleusSolidityFeatureId = pdc.objFeatureIds[nucleusSolidity]
    mask = (pdc.objFeatures[: , nucleusSolidityFeatureId] > minNucSolidity)
    if pdc.objQualityControlFeatureId >= 0:
        failMask = numpy.invert(mask)
        qualityControlFeature[numpy.logical_and(validCellMask, failMask)] = QUALITY_CONTROL_TOO_SMALL_NUCLEUS_SOLIDITY
    validCellMask = numpy.logical_and(validCellMask, mask)
    n7 = validCellMask.sum()
    print 'found %d cells with minimum nucleus solidity' % (n6 - n7)

    qualityControlFeature[validCellMask] = QUALITY_CONTROL_VALID
    pdc.objFeatures[: , pdc.objQualityControlFeatureId] = qualityControlFeature

    if numOfProcesses > 1:
        d, validImageMask, validCellMask \
            = quality_control_mp.quality_control_images(
                pdc, validImageMask, validCellMask,
                nprocesses=numOfProcesses)

    else:

        d = {}

        #print 'checking %d images' % len(pdc.images)

        for image in pdc.images:

            sys.stdout.write('\rchecking {} of {} images'.format(
                image.index + 1, len(pdc.images)))
            sys.stdout.flush()

            imgCellMask = pdc.objFeatures[: , pdc.objImageFeatureId] == image.index

            if image.state != 'ok':
                pass

            # minimum number of cells
            elif validCellMask[imgCellMask].sum() < minCells:
                image.state = 'not_enough_ok_cells'
                if pdc.imgQualityControlFeatureId >= 0:
                    pdc.imgFeatures[image.index, pdc.imgQualityControlFeatureId] = QUALITY_CONTROL_NOT_ENOUGH_VALID_CELLS

            # maximum number of cells
            elif validCellMask[imgCellMask].sum() > maxCells:
                #print 'too_many_cells'
                image.state = 'too_many_cells'
                if pdc.imgQualityControlFeatureId >= 0:
                    pdc.imgFeatures[image.index, pdc.imgQualityControlFeatureId] = QUALITY_CONTROL_TOO_MANY_CELLS

            else:
                # minimal number of background pixels
                cellFeatureId = pdc.objFeatureIds[cellArea]
                areaOccupiedByCells = sum(pdc.objFeatures[imgCellMask , cellFeatureId])
                if (imageWidth * imageHeight - areaOccupiedByCells < minAreaBg):
                    #print 'not_enough_bg_Pixels for image(%d): %s' % (image.index,image.name)
                    image.state = 'not_enough_bg_pixels'
                    if pdc.imgQualityControlFeatureId >= 0:
                        pdc.imgFeatures[image.index, pdc.imgQualityControlFeatureId] = QUALITY_CONTROL_NOT_ENOUGH_BG_PIXELS

            if image.state != 'ok':
                if not d.has_key(image.state):
                    d[image.state] = [0, 0]
                d[image.state][0] += 1
                d[image.state][1] += numpy.sum(numpy.logical_and(validCellMask, imgCellMask))
                validImageMask[image.index] = False
                validCellMask[imgCellMask] = False

        sys.stdout.write('\n')
        sys.stdout.flush()

    for k,v in d.iteritems():
        v1,v2 = v
        print 'found %d images containing %d cells with state "%s"' % (v1,v2,k)

    return validImageMask, validCellMask
