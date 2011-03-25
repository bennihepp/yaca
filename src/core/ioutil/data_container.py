import numpy



OBJECT_ID_FEATURE_NAME = 'OBJECT_ID'
TREATMENT_ID_FEATURE_NAME = 'TREATMENT_ID'
REPLICATE_ID_FEATURE_NAME = 'REPLICATE_ID'
IMAGE_ID_FEATURE_NAME = 'IMAGE_ID'
STATE_ID_FEATURE_NAME = ''

YACA_DATA_VERSION = '0.2'



# definition of the data structure used for the CellProfiler2 output
class yaca_data_container(object):

    __version__ = YACA_DATA_VERSION

    VERSION_ATTR_DICT = {
        0.1 : [ 'images', 'objects', 'treatments', 'treatmentByName',
                'imgFeatureIds', 'imgImageFeatureId', 'imgTreatmentFeatureId',
                'objFeatureIds', 'objObjectFeatureId', 'objImageFeatureId', 'objTreatmentFeatureId',
                'errors' ],
        0.2 : [ 'images', 'objects', 'treatments', 'treatmentByName', 'replicates', 'replicateByName',
                'imgFeatureIds', 'imgImageFeatureId', 'imgTreatmentFeatureId',
                'objFeatureIds', 'objObjectFeatureId', 'objImageFeatureId', 'objTreatmentFeatureId',
                'errors' ],
    }

    def __init__(self):

        self.images = []
        self.objects = []

        self.treatments = []
        self.treatmentByName = {}

        #self.replicates = []
        #self.replicateByName = {}

        self.imgFeatureIds = {}
        self.imgFeatures = None #numpy.array(0.0)

        self.imgImageFeatureId = -1
        self.imgTreatmentFeatureId = -1
        self.imgReplicateFeatureId = -1

        self.objFeatureIds = {}
        self.objFeatures = None #numpy.array(0.0)

        self.objObjectFeatureId = -1
        self.objImageFeatureId = -1
        self.objTreatmentFeatureId = -1
        self.objReplicateFeatureId = -1

        self.errors = []

    def export_container(self):
        d = {}
        for attr_name in VERSION_ATTR_DICT[ self.__version__ ]:
            d[ attr_name ] = self.__getattr( attr_name )
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
        if version not in VERSION_ATTR_DICT:
            raise Exception( 'Unknown YACA Data Container version: %s' % version )
        self.__version__ = version
        for attr_name in VERSION_ATTR_DICT[ version ]:
            self.__setattr( attr_name, container[ attr_name ] )

class yaca_data_image(object):

    def __init__(self):
        self.rowId = -1
        self.import_state = 'ok'
        self.state = None
        self.treatment = None
        self.replicate = None
        self.imageFiles = []
        self.properties = {}
        self.cellMask = None
#        self.objIds = None #numpy.array(0)


class yaca_data_object(object):

    def __init__(self):
        self.rowId = -1
        self.import_state = 'ok'
        self.state = None
        self.image = None
        self.position_x = -1.0
        self.position_y = -1.0
#        self.properties = {}


class yaca_data_treatment(object):

    def __init__(self, name):
        self.rowId = -1
        self.name = name
        self.imgMask = None
        self.cellMask = None
#        self.imgIds = None #numpy.array(0)
#        self.objIds = None #numpy.array(0)


class yaca_data_replicate(object):

    def __init__(self, name):
        self.rowId = -1
        self.name = name
        self.imgMask = None
        self.cellMask = None
#        self.imgIds = None #numpy.array(0)
#        self.objIds = None #numpy.array(0)


class yaca_data_error(object):

    def __init__(self, exception, traceback, entity):
        self.exception = exception
        self.traceback = traceback
        self.entity = entity
