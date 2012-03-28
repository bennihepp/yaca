# -*- coding: utf-8 -*-

"""
quality_control_mp.py -- Quality control of images and cells with
                         multiprocessing support.

See the documentation for further details.
"""

# This software is distributed under the FreeBSD License.
# See the accompanying file LICENSE for details.
#
# Copyright 2011 Benjamin Hepp

import sys
import multiprocessing

import numpy

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


# quality control of the data
def quality_control_images(pdc, validImageMask, validCellMask, nprocesses=10):

    print('preparing multiple processes...'

    shared_objectFeatures = multiprocessing.Array(
        'd',
        pdc.objFeatures.reshape(
            (pdc.objFeatures.shape[0] * pdc.objFeatures.shape[1])
        ),
        lock=False)
    shared_validCellMask = multiprocessing.Array(
        'L', validCellMask, lock=False)

    queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()
    pool = []
    objFeatures = pdc.objFeatures
    imgFeatures = pdc.imgFeatures
    objects = pdc.objects
    images = pdc.images
    pdc.objFeatures = None
    pdc.imgFeatures = None
    pdc.objects = None
    pdc.images = None
    args = (queue, result_queue,
            pdc,
            imgFeatures.shape[0],
            objFeatures.shape[0],
            shared_objectFeatures,
            shared_validCellMask)
    for n in xrange(nprocesses):
        p = multiprocessing.Process(
            target=quality_control_images_worker,
            args=args)
        pool.append(p)
        p.start()
    pdc.objFeatures = objFeatures
    pdc.imgFeatures = imgFeatures
    pdc.objects = objects
    pdc.images = images

    i = 0
    for image in pdc.images:
        if image.state == 'ok':
            queue.put(image)
        else:
            i += 1

    for n in xrange(len(pool)):
        queue.put('QUIT')

    n = 0
    d = {}
    while n < len(pool):
        try:
            sys.stdout.write('\rchecking {} of {} images'.format(
                i, len(pdc.images)))
            sys.stdout.flush()
            obj = result_queue.get()
        except IOError as e:
            # this is a dirty hack but I don't see another way to
            # use this with PyQt4
            if e.errno == 4:
                continue
            else:
                raise

        if obj == 'QUIT':
            n += 1
        else:
            image, qc_feature, obj_count, imgCellMask = obj
            if qc_feature is not None:
                pdc.images[image.index].state = image.state
                if qc_feature is not None \
                   and pdc.imgQualityControlFeatureId >= 0:
                    pdc.imgFeatures[image.index, pdc.imgQualityControlFeatureId] \
                        = qc_feature
                if image.state != 'ok':
                    if not image.state in d:
                        d[image.state] = [0, 0]
                    d[image.state][0] += 1
                    d[image.state][1] += obj_count
                    validImageMask[image.index] = False
                    validCellMask[imgCellMask] = False
            i += 1

    sys.stdout.write('\n')
    sys.stdout.flush()

    for p in pool:
        p.join()

    queue.close()
    result_queue.close()

    return d, validImageMask, validCellMask


def quality_control_images_worker(queue, result_queue,
                                  dummy_pdc, num_of_images,
                                  num_of_objects,
                                  shared_objectFeatures,
                                  shared_validCellMask):
    from quality_control import minCells, maxCells, cellArea, imageWidth, \
         imageHeight, minAreaBg

    objectFeatures = numpy.frombuffer(shared_objectFeatures, dtype=numpy.float)
    objectFeatures = objectFeatures.reshape(
        (num_of_objects, len(shared_objectFeatures) / num_of_objects))
    validCellMask = numpy.frombuffer(shared_validCellMask, dtype=numpy.int)
    validCellMask = numpy.asarray(validCellMask, dtype=numpy.bool)

    while True:

        obj = queue.get()
        if obj == 'QUIT':
            result_queue.put('QUIT')
            break

        else:

            image = obj

            imgCellMask \
                = objectFeatures[:, dummy_pdc.objImageFeatureId] == image.index

            obj_count \
                = numpy.sum(numpy.logical_and(validCellMask, imgCellMask))

            # minimum number of cells
            if validCellMask[imgCellMask].sum() < minCells:
                image.state = 'not_enough_ok_cells'
                result_queue.put(
                    (image, QUALITY_CONTROL_NOT_ENOUGH_VALID_CELLS, obj_count,
                     imgCellMask))

            # maximum number of cells
            elif validCellMask[imgCellMask].sum() > maxCells:
                image.state = 'too_many_cells'
                result_queue.put(
                    (image, QUALITY_CONTROL_TOO_MANY_CELLS, obj_count,
                     imgCellMask))

            else:
                # minimal number of background pixels
                cellFeatureId = dummy_pdc.objFeatureIds[cellArea]
                areaOccupiedByCells \
                    = sum(objectFeatures[imgCellMask, cellFeatureId])
                areaBg = imageWidth * imageHeight - areaOccupiedByCells
                if (areaBg < minAreaBg):
                    image.state = 'not_enough_bg_pixels'
                    result_queue.put(
                        (image, QUALITY_CONTROL_NOT_ENOUGH_BG_PIXELS,
                         obj_count,
                         imgCellMask))
                else:
                    result_queue.put((image, None, None, None))
