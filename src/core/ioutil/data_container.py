import numpy



OBJECT_ID_FEATURE_NAME = 'OBJECT_ID'
TREATMENT_ID_FEATURE_NAME = 'TREATMENT_ID'
IMAGE_ID_FEATURE_NAME = 'IMAGE_ID'
STATE_ID_FEATURE_NAME = ''

YACA_DATA_VERSION = '0.1'



# definition of the data structure used for the CellProfiler2 output
class yaca_data_container(object):

    __version__ = YACA_DATA_VERSION

    def __init__(self):

        self.images = []
        self.objects = []

        self.treatments = []
        self.treatmentByName = {}

        self.imgFeatureIds = {}
        self.imgFeatures = None #numpy.array(0.0)

        self.imgImageFeatureId = -1
        self.imgTreatmentFeatureId = -1

        self.objFeatureIds = {}
        self.objFeatures = None #numpy.array(0.0)

        self.objObjectFeatureId = -1
        self.objImageFeatureId = -1
        self.objTreatmentFeatureId = -1

        self.errors = []

class yaca_data_image(object):

    def __init__(self):
        self.rowId = -1
        self.import_state = 'ok'
        self.state = None
        self.treatment = None
        self.imageFiles = []
        self.properties = {}
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
#        self.imgIds = None #numpy.array(0)
#        self.objIds = None #numpy.array(0)


class yaca_data_error(object):

    def __init__(self, exception, traceback, entity):
        self.exception = exception
        self.traceback = traceback
        self.entity = entity
