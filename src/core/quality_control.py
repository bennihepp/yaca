import sys
import numpy



import parameter_utils as utils

__dict__ = sys.modules[ __name__ ].__dict__

utils.register_module( __name__, 'Quality control of cells and images', __dict__ )

import importer
utils.add_required_state( __name__, importer.__name__, 'imported' )

utils.register_parameter( __name__, 'positionX', utils.PARAM_OBJ_FEATURE, 'Feature ID for the x coordinate of a cell (in pixel)', 'nucleus_Location_Center_X' )
utils.register_parameter( __name__, 'positionY', utils.PARAM_OBJ_FEATURE, 'Feature ID for the y coordinate of a cell (in pixel)', 'nucleus_Location_Center_Y')

utils.register_parameter( __name__, 'cellArea', utils.PARAM_OBJ_FEATURE, 'Feature ID for the area of a cell (in pixel)', 'cell_AreaShape_Area' )
utils.register_parameter( __name__, 'nucleusArea', utils.PARAM_OBJ_FEATURE, 'Feature ID for the area of a nucleus (in pixel)', 'nucleus_AreaShape_Area' )

utils.register_parameter( __name__, 'nucleusSolidity', utils.PARAM_OBJ_FEATURE, 'Feature ID for the solidity of a nucleus', 'nucleus_AreaShape_Solidity' )

utils.register_parameter( __name__, 'minAreaBg', utils.PARAM_INT, 'Minimum number of background pixels', 50000, 0, None )
utils.register_parameter( __name__, 'minNucArea', utils.PARAM_INT, 'Minimum area of nuclei in pixels', 1000, 0, None )
utils.register_parameter( __name__, 'maxNucArea', utils.PARAM_INT, 'Maximum area of nuclei in pixels', 4000, 0, None )

utils.register_parameter( __name__, 'minNucSolidity', utils.PARAM_FLOAT, 'Minimum nuclei solidity', 0.9, 0.0, 1.0 )

utils.register_parameter( __name__, 'minDistLeft', utils.PARAM_INT, 'Minimum distance of cells to left image margin', 100, 0, None )
utils.register_parameter( __name__, 'minDistRight', utils.PARAM_INT, 'Minimum distance of cells to right image margin', 100, 0, None )
utils.register_parameter( __name__, 'minDistTop', utils.PARAM_INT, 'Minimum distance of cells top image margin', 100, 0, None )
utils.register_parameter( __name__, 'minDistBottom', utils.PARAM_INT, 'Minimum distance of cells to bottom image margin', 100, 0, None )

utils.register_parameter( __name__, 'imageWidth', utils.PARAM_INT, 'Width of images in pixel', 1344, 1, None )
utils.register_parameter( __name__, 'imageHeight', utils.PARAM_INT, 'Height of images in pixel', 1024, 1, None )

utils.register_parameter( __name__, 'minCyToNucAreaFrac', utils.PARAM_FLOAT, 'Minimum allowed fraction of cytoplasma to nucleus area', 1.5, 0.0, None )
utils.register_parameter( __name__, 'maxCyToNucAreaFrac', utils.PARAM_FLOAT, 'Maximum allowed fraction of cytoplasma to nucleus area', 15, 0.0, None )

utils.register_parameter( __name__, 'minCells', utils.PARAM_INT, 'Minimum number of cells per image', 1, 1, None)
utils.register_parameter( __name__, 'maxCells', utils.PARAM_INT, 'Maximum number of cells per image', 300, 1, None )



# quality control of the data
def quality_control( pdc ):

    print 'cellArea: %s' % cellArea
    print 'nucleusArea: ', nucleusArea
    print 'positionX: ', positionX
    print 'positionY: ', positionY

    validImageMask = numpy.empty( ( len(pdc.images), ) , dtype=bool )
    validImageMask[:] = True

    validCellMask = numpy.empty( ( len(pdc.objects), ) , dtype=bool )
    validCellMask[:] = True


    # cells are at image periphery
    #print "\n".join( pdc.objFeatureIds.keys() )
    featureId = pdc.objFeatureIds[ positionX ]
    n1 = validCellMask.sum()
    validCellMask = numpy.logical_and(validCellMask,
                ( pdc.objFeatures[ : , featureId ] > minDistLeft )
    )
    validCellMask = numpy.logical_and(validCellMask,
                ( pdc.objFeatures[ : , featureId ] < imageWidth-minDistRight )
    )
    featureId = pdc.objFeatureIds[ positionY ]
    validCellMask = numpy.logical_and(validCellMask,
                ( pdc.objFeatures[ : , featureId ] > minDistTop )
    )
    validCellMask = numpy.logical_and(validCellMask,
                ( pdc.objFeatures[ : , featureId ] < imageHeight-minDistBottom )
    )
    n2 = validCellMask.sum()
    print 'found %d cells out of image periphery' % (n1 - n2)

    # minimum nucleus area
    nucleusFeatureId = pdc.objFeatureIds[ nucleusArea ]
    #thresholdNucleusArea = numpy.median(image.objFeatures[:,featureId] \
    #              - 2 * numpy.mean( numpy.abs(
    #                      image.objFeatures[:,featureId]
    #                      - numpy.mean(image.objFeatures[:,featureId])
    #              ) )
    #)
    validCellMask = numpy.logical_and(validCellMask,
                ( pdc.objFeatures[ : , nucleusFeatureId ] > minNucArea )
    )
    n3 = validCellMask.sum()
    print 'found %d cells with minimum nucleus area' % (n2 - n3)

    # maximum nucleus area
    validCellMask = numpy.logical_and(validCellMask,
                ( pdc.objFeatures[ : , nucleusFeatureId ] < maxNucArea )
    )
    n4 = validCellMask.sum()
    print 'found %d cells with minimum nucleus area' % (n3 - n4)

    # minimum relative cytoplasm area
    #cellFeatureId = pdc.cellFeatureIds['AreaShape_Area']
    cellFeatureId = pdc.objFeatureIds[ cellArea ]
    validCellMask = numpy.logical_and(validCellMask,
                pdc.objFeatures[ : , cellFeatureId ]
                          > ( minCyToNucAreaFrac * pdc.objFeatures[ : , nucleusFeatureId ] )
    )
    n5 = validCellMask.sum()
    print 'found %d cells with maximum relative cytoplasm area' % (n4 - n5)

    # maximum relative cytoplasm area
    validCellMask = numpy.logical_and(validCellMask,
                ( pdc.objFeatures[ : , cellFeatureId ]
                  < maxCyToNucAreaFrac * pdc.objFeatures[ : , nucleusFeatureId ] )
    )
    n6 = validCellMask.sum()
    print 'found %d cells with maximum relative cytoplasm area' % (n5 - n6)

    # minimum nucleus solidity
    nucleusSolidityFeatureId = pdc.objFeatureIds[ nucleusSolidity ]
    validCellMask = numpy.logical_and(validCellMask,
                ( pdc.objFeatures[ : , nucleusSolidityFeatureId ] > minNucSolidity )
    )
    n7 = validCellMask.sum()
    print 'found %d cells with minimum nucleus solidity' % (n6 - n7)

    d = {}

    print 'checking %d images' % len( pdc.images )

    for image in pdc.images:

        imgCellMask = pdc.objFeatures[ : , pdc.objImageFeatureId ] == image.rowId

        if image.state != 'ok':
            pass

        # minimum number of cells
        elif validCellMask[ imgCellMask ].sum() < minCells:
            image.state = 'not_enough_ok_cells'

        # maximum number of cells
        elif validCellMask[ imgCellMask ].sum() > maxCells:
            #print 'too_many_cells'
            image.state = 'too_many_cells'

        else:
            # minimal number of background pixels
            cellFeatureId = pdc.objFeatureIds[ cellArea ]
            areaOccupiedByCells = sum( pdc.objFeatures[ imgCellMask , cellFeatureId ] )
            if (imageWidth * imageHeight - areaOccupiedByCells < minAreaBg):
                #print 'not_enough_bg_Pixels for image(%d): %s' % (image.rowId,image.name)
                image.state = 'not_enough_bg_pixels'

        if image.state != 'ok':
            if not d.has_key( image.state ):
                d[ image.state ] = 0
            d[ image.state ] += 1
            validImageMask[ image.rowId ] = False
            validCellMask[ imgCellMask ] = False

    for k,v in d.iteritems():
        print 'found %d images with state "%s"' % (v,k)

    return validImageMask, validCellMask
