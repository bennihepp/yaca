# -*- coding: utf-8 -*-

"""
data_container.py -- Definition of the data container class.
"""

# This software is distributed under the FreeBSD License.
# See the accompanying file LICENSE for details.
# 
# Copyright 2011 Benjamin Hepp

import numpy

OBJECT_ID_FEATURE_NAME = 'OBJECT_ID'
WELL_ID_FEATURE_NAME = 'WELL_ID'
TREATMENT_ID_FEATURE_NAME = 'TREATMENT_ID'
REPLICATE_ID_FEATURE_NAME = 'REPLICATE_ID'
IMAGE_ID_FEATURE_NAME = 'IMAGE_ID'
QUALITY_CONTROL_FEATURE_NAME = 'Quality_Control'

YACA_DATA_VERSION = '0.4'

# definition of the data structure used for the CellProfiler2 output
class yaca_data_container(object):

    __version__ = YACA_DATA_VERSION

    VERSION_ATTR_DICT = {
        '0.1' : [ 'images', 'objects', 'treatments', 'treatmentByName',
                'imgFeatureIds', 'imgImageFeatureId', 'imgTreatmentFeatureId',
                'objFeatureIds', 'objObjectFeatureId', 'objImageFeatureId', 'objTreatmentFeatureId',
                'errors' ],
        '0.2' : [ 'images', 'objects', 'treatments', 'treatmentByName', 'replicates', 'replicateByName',
                'imgFeatureIds', 'imgImageFeatureId', 'imgTreatmentFeatureId', 'imgReplicateFeatureId',
                'objFeatureIds', 'objObjectFeatureId', 'objImageFeatureId', 'objTreatmentFeatureId', 'objReplicateFeatureId',
                'errors' ],
        '0.3' : [ 'images', 'objects',
                'wells', 'wellByName',
                'treatments', 'treatmentByName',
                'replicates', 'replicateByName',
                'imgFeatureIds', 'imgImageFeatureId', 'imgWellFeatureId', 'imgTreatmentFeatureId', 'imgReplicateFeatureId',
                'objFeatureIds', 'objObjectFeatureId', 'objImageFeatureId', 'objWellFeatureId', 'objTreatmentFeatureId', 'objReplicateFeatureId',
                'errors' ],
        '0.4' : [ 'images', 'objects',
                'wells', 'wellByName',
                'treatments', 'treatmentByName',
                'replicates', 'replicateByName',
                'imgFeatureIds', 'imgImageFeatureId', 'imgWellFeatureId', 'imgTreatmentFeatureId', 'imgReplicateFeatureId', 'imgQualityControlFeatureId',
                'objFeatureIds', 'objObjectFeatureId', 'objImageFeatureId', 'objWellFeatureId', 'objTreatmentFeatureId', 'objReplicateFeatureId', 'objQualityControlFeatureId',
                'errors' ]
    }

    def __init__(self):

        self.images = []
        self.objects = []

        self.wells = []
        self.wellByName = {}

        self.treatments = []
        self.treatmentByName = {}

        self.replicates = []
        self.replicateByName = {}

        self.imgFeatureIds = {}
        self.imgFeatures = None #numpy.array(0.0)

        self.imgImageFeatureId = -1
        self.imgWellFeatureId = -1
        self.imgTreatmentFeatureId = -1
        self.imgReplicateFeatureId = -1
        self.imgQualityControlFeatureId = -1

        self.objFeatureIds = {}
        self.objFeatures = None #numpy.array(0.0)

        self.objObjectFeatureId = -1
        self.objImageFeatureId = -1
        self.objWellFeatureId = -1
        self.objTreatmentFeatureId = -1
        self.objReplicateFeatureId = -1
        self.objQualityControlFeatureId = -1

        self.errors = []

    def objFeatureName(self, fid, listOfNames=False):
        names = filter( ( lambda (name,id): id == fid ), self.objFeatureIds.iteritems() )
        if listOfNames:
            return zip( *names )[0]
        else:
            if len( names ) == 0:
                raise Exception( 'No such feature: %d' % fid )
            return names[0][0]

    def export_container(self):
        d = { '__version__' : self.__version__}
        for attr_name in yaca_data_container.VERSION_ATTR_DICT[ self.__version__ ]:
            d[ attr_name ] = self.__getattribute__( attr_name )
        return d

    @staticmethod
    def import_container(container):
        if type( container ) == dict:
            ydc = yaca_data_container()
            ydc.__import_container( container )
        else:
            ydc = container
        return ydc

    def __import_container(self, container):
        version = container[ '__version__' ]
        if version not in yaca_data_container.VERSION_ATTR_DICT:
            raise Exception( 'Unknown YACA Data Container version: %s' % version )
        self.__version__ = version
        for attr_name in yaca_data_container.VERSION_ATTR_DICT[ version ]:
            self.__setattr__( attr_name, container[ attr_name ] )

class yaca_data_image(object):

    def __init__(self):
        self.index = -1
        self.import_state = 'ok'
        self.state = None
        self.treatment = None
        self.replicate = None
        self.well = None
        self.imageFiles = []
        self.properties = {}
        #self.cellMask = None
#        self.objIds = None #numpy.array(0)


class yaca_data_object(object):

    def __init__(self):
        self.index = -1
        self.import_state = 'ok'
        self.state = None
        self.image = None
        self.position_x = -1.0
        self.position_y = -1.0
#        self.properties = {}


class yaca_data_well(object):

    def __init__(self, name):
        self.index = -1
        self.name = name
        #self.imgMask = None
        #self.cellMask = None
#        self.imgIds = None #numpy.array(0)
#        self.objIds = None #numpy.array(0)


class yaca_data_treatment(object):

    def __init__(self, name):
        self.index = -1
        self.name = name
        #self.imgMask = None
        #self.cellMask = None
#        self.imgIds = None #numpy.array(0)
#        self.objIds = None #numpy.array(0)


class yaca_data_replicate(object):

    def __init__(self, name):
        self.index = -1
        self.name = name
        #self.imgMask = None
        #self.cellMask = None
#        self.imgIds = None #numpy.array(0)
#        self.objIds = None #numpy.array(0)


class yaca_data_error(object):

    def __init__(self, exception, traceback, entity):
        self.exception = exception
        self.traceback = traceback
        self.entity = entity
